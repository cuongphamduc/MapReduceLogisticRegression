import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.*;

/*
arr[0] : số lượng feature
arr[1] : learing rate
arr[2] : số lần lặp
arr[3] : input
arr[4] : output
arr[5] : test
 */

public class Run {
    public static int num_features;
    public static float lr;

    public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
        File file = new File("F:\\log\\log.txt");

        if (!file.exists()) {
            file.createNewFile();
        }

        FileWriter fw = new FileWriter(file.getAbsoluteFile(), true);
        BufferedWriter bw = new BufferedWriter(fw);

        num_features = Integer.parseInt(args[0]);
        lr = Float.parseFloat(args[1]);

        Configuration conf = new Configuration();
        FileSystem hdfs = FileSystem.get(conf);
        Float[] theta = new Float[num_features + 10];

        for (int i = 1; i <= Integer.parseInt(args[2]); i++) {
            if (i == 1) {
                for (int j = 1; j <= num_features; j++) {
                    theta[j] = 0.0f;
                }
            } else {

                BufferedReader br1 = new BufferedReader(new InputStreamReader(hdfs.open(new Path(args[4] + "/part-r-00000"))));
                String line1;

                while ((line1 = br1.readLine()) != null) {
                    String[] tmp = line1.split("\t");
                    int index = Integer.parseInt(tmp[0].substring(5));
                    theta[index] = Float.parseFloat(tmp[1]);
                }

                br1.close();
            }

            if (hdfs.exists(new Path(args[4]))) {
                hdfs.delete(new Path(args[4]), true);
            }

            conf.setFloat("lr", lr);
            conf.setInt("numfe", num_features);

            for (int j = 1; j <= num_features; j++) {
                conf.setFloat("theta".concat(String.valueOf(j)), theta[j]);
            }
            Job job = Job.getInstance(conf, "Calculation of Theta");
            job.setJarByClass(Run.class);

            FileInputFormat.setInputPaths(job, new Path(args[3]));
            FileOutputFormat.setOutputPath(job, new Path(args[4]));

            job.setMapperClass(logisMap.class);
            job.setReducerClass(logisReduce.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(FloatWritable.class);
            job.waitForCompletion(true);

            BufferedReader test = new BufferedReader(new InputStreamReader(hdfs.open(new Path(args[5] + "/dota2Test.csv"))));
            String s;

            int correct = 0, total = 0;
            Integer[] Xi = new Integer[1000];

            while ((s = test.readLine()) != null) {
                String[] tmp = s.split("\\,");
                for (int j = 1; j < tmp.length; j++) {
                    Xi[j] = Integer.parseInt(tmp[j]);
                }

                float sum = 0;

                for (int j = 1; j < tmp.length; j++) {
                    sum += (Xi[i] * theta[i]);
                }

                float predict = (float) (1 / (1 + (Math.exp(-sum))));

                int Yi = Integer.parseInt(tmp[0]);

                if ((predict >= 0.5 && Yi == 1) || (predict < 0.5 && Yi == -1)) {
                    correct++;
                }

                total++;
            }

            test.close();
            bw.write("Accuracy of loop " + i + " is : " + (float) correct/total + "\n");

        }

        hdfs.close();
        bw.close();
    }
}
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;


public class logisMap extends Mapper<LongWritable, Text, Text, FloatWritable> {

    public static int count = 0;
    public static float lr = 0.0f;
    public static Integer[] Xi = null;
    public static Float[] theta = null;
    public static int num_features = 0;

    @Override
    public void setup(Context context) {
        lr = context.getConfiguration().getFloat("lr", 0.0f);
        num_features = context.getConfiguration().getInt("numfe", 0);
        Xi = new Integer[num_features + 10];
        theta = new Float[num_features + 10];
    }

    public void map(LongWritable key, Text value, Context context) {
        ++count;
        String[] temp = value.toString().split("\\,");

        if (count == 1) {
            for (int i = 1; i <= num_features; i++) {
                theta[i] = context.getConfiguration().getFloat("theta".concat(String.valueOf(i)), 0.0f);
            }
        }

        for (int i = 1; i <= num_features; i++) {
            Xi[i] = Integer.parseInt(temp[i]);
        }

        float exp = 0;

        for (int i = 1; i <= num_features; i++) {
            exp += (Xi[i] * theta[i]);
        }

        float predict = (float) (1 / (1 + (Math.exp(-exp))));

        int Yi = Integer.parseInt(temp[0]);

        if (Yi == -1) {
            Yi = 0;
        }

        for (int i = 1; i <= num_features; i++) {
            float update = theta[i] + lr * (Yi - predict) * (Xi[i]);
            theta[i] = update;
        }
    }

    @Override
    public void cleanup(Context context) throws IOException, InterruptedException {
        for (int i = 1; i <= num_features; i++) {
            context.write(new Text("theta" + i), new FloatWritable(theta[i]));
        }
    }
}

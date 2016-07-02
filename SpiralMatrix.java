package attune.platform.source.log.view;

/**
 * Created by sudnya on 6/16/16.
 */
public class SpiralMatrix {
    public static void printSpiral(int[][] matrix, int r, int c) {
        int top = 0;
        int down = matrix.length - 1;
        int left = 0;
        int right = matrix[0].length - 1;

        while(true)
        {
            // Print top row
            for(int j = left; j <= right; ++j) {
                System.out.print(matrix[top][j] + " ");
            }
            top++;

            if(top > down || left > right)
                break;
            //Print the rightmost column
            for(int i = top; i <= down; ++i) {
                System.out.print(matrix[i][right] + " ");
            }
            right--;

            if(top > down || left > right)
                break;
            //Print the bottom row
            for(int j = right; j >= left; --j) {
                System.out.print(matrix[down][j] + " ");
            }
            down--;
            if(top > down || left > right)
                break;
            //Print the leftmost column
            for(int i = down; i >= top; --i) {
                System.out.print(matrix[i][left] + " ");
            }
            left++;
            if(top > down || left > right)
                break;
        }
    }

    public static void main(String args[] ) throws Exception {
        /* Enter your code here. Read input from STDIN. Print output to STDOUT */
        int rows = Integer.parseInt(args[0]);
        int cols = Integer.parseInt(args[1]);
        int [][] a = new int[rows][cols];

        for (int i = 0; i < rows; ++i) {
            String r = args[i+2];
            String[] x = r.split(",");

            for (int j = 0; j < x.length; ++j) {
                a[i][j] = Integer.parseInt(x[j]);
            }
        }
        printSpiral(a, rows, cols);

    }
}

package attune.platform.source.log.view;

/**
 * Created by sudnya on 6/16/16.
 */
public class NumberOfPathsInMxNMatrix {

    static int numberOfPaths(int[][] T) {
        int clamp = (int)Math.pow(10, 9) + 7;
        int col = T[0].length;
        int row = T.length;

        for (int i=1; i < row; i++) {
            for (int j=1; j < col; j++) {
                if (T[i][j] == 1) {
                    T[i][j] = T[i - 1][j] + T[i][j - 1];
                } else {
                    T[i][j] = 0;
                }
            }
        }
        return (T[row-1][col-1])%clamp;
    }


    public int countPathsRecursive(int n, int m){
        if(n == 1 || m == 1){
            return 1;
        }
        return countPathsRecursive(n-1, m) + countPathsRecursive(n, m-1);
    }

    public static void main(String args[]){
        int[][] a = {{1, 0, 0, 0}, {1, 1, 1, 1}, {1, 0, 1, 1}};
        //int[][] a = {{1, 1,1,1}, {1, 1, 1, 1}, {1, 1, 1, 1}};
        //int[][] a = {{0,0,0,0}, {0,0,0,0}, {0,0,0,0}};

        NumberOfPathsInMxNMatrix nop = new NumberOfPathsInMxNMatrix();
        System.out.println(nop.numberOfPaths(a));
        //System.out.print(nop.countPathsRecursive(3,3));
    }

};

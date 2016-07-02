package attune.platform.source.log.view;

import java.io.IOException;

/**
 * Created by sudnya on 6/16/16.
 */
public class StringSimilarity {

    static int getScore(String s1, int off) {
        int retVal = 0;
        StringBuilder s = new StringBuilder(s;)
        //ensure s1 is always > s2
        for (int i = 0; i < s1.length()-off; ++i) {
            if (s1.charAt(i) != s1.charAt(i+off))
                break;
            retVal++;
        }
        return retVal;
    }

    static int getMatchingScore(String s, int maxL) {
        int retVal = 0;

        for (int i = 0; i < maxL; ++i) {
            retVal += getScore(s, i);
        }

        return retVal;
    }

    static int[] StringSimilarity(String[] inputs) {
        int numberOfStrings = inputs.length;
        if (numberOfStrings < 1 || numberOfStrings > 10) {
            return new int[1];
        }

        int[] retVal = new int[numberOfStrings];

        for (int i = 0; i < numberOfStrings; ++i ) {
            int strLen = inputs[i].toString().length();
            if (strLen > 100000) {
                retVal[i] = -1;
                break;
            }

            retVal[i] = getMatchingScore(inputs[i].toString(), strLen);
        }

        return retVal;

    }
    /*
       static int[] StringSimilarity(String[] inputs) {
        int numberOfStrings = Integer.parseInt(inputs[0]);
        int[] retVal = new int[numberOfStrings];

        for (int i = 0; i < numberOfStrings; ++i ) {
            retVal[i] = getMatchingScore(inputs[i+1]);
        }

        return retVal;

    }
     */

    public static void main(String args[]) throws IOException {
        String[] inputs = {"ababaa", "aa"};
        //String[] inputs = {"1", "ababaabbbbasjdhahgfjdhfgjdshgfjdhsgf"};
        int [] scores = StringSimilarity.StringSimilarity(inputs);
        for (int i = 0; i < scores.length; ++ i) {
            System.out.println("Score for i: " + i + " is " + scores[i]);
        }
    }
};

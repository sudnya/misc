package attune.platform.source.log.view;

import java.io.IOException;

/**
 * Created by sudnya on 6/16/16.
 */
public class LandingStep {
    static int maxStep(int N, int K) {
        if (N < 1 || N > 2000) {
            assert (false);
            return 0;
        }

        if (K < 1 || K > 4000000) {
            assert (false);
            return 0;
        }

        int currentStep = 0;
        //int landingStep = 0;

        for (int i = 1; i <= N; ++i) {

            //landingStep = currentStep + i;
            if (currentStep + i == K)
                continue;
            else
                currentStep += i;
            System.out.println("i: " + i + " current " + currentStep );//+ " landing " + landingStep );

            /*if (landingStep == K) {
                System.out.println("landing step == K :" + K + " keeping current: " + currentStep);
                continue;
            } else {

                currentStep = landingStep;
            }*/
        }
        return currentStep;
    }

    public static void main(String[] args) throws IOException {
        /*int m = LandingStep.maxStep(2,2);
        System.out.println("MaxStep " + m);
        m = LandingStep.maxStep(2, 1);
        System.out.println("MaxStep " + m);
        m = LandingStep.maxStep(9, 7);
        System.out.println("MaxStep " + m);*/
        //int m = LandingStep.maxStep(9, 21);
        int m = LandingStep.maxStep(9, 36 );
        System.out.println("MaxStep " + m);

    }

};

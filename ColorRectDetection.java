import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

public class ColorRectDetection {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) {
        // camera set up
        VideoCapture capture = new VideoCapture(1); // used to change what camera you are using, cnage it whenver using indexes
        if (!capture.isOpened()) {
            System.out.println("Cannot open the camera!");
            return;
        }
        //resolution
        capture.set(Videoio.CAP_PROP_FRAME_WIDTH, 640);
        capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 480);

        Mat frame = new Mat();

        //processing 
        while (true) {
            if (!capture.read(frame)) {
                System.out.println("Failed to capture image");
                break;
            }
            //represent images and matrices
            Mat hsv = new Mat();
            Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_BGR2HSV);

            // detects blocks and their color
            detectColor(hsv, frame, new Scalar(0, 100, 100), new Scalar(10, 255, 255), "Red");
            detectColor(hsv, frame, new Scalar(40, 70, 70), new Scalar(80, 255, 255), "Green");
            detectColor(hsv, frame, new Scalar(100, 150, 0), new Scalar(140, 255, 255), "Blue");

            // show results
            Imgproc.imshow("Color Rect Detection", frame);
            if (Imgproc.waitKey(1) == 27) break; 
        }
        // cleanup
        capture.release();
        Imgproc.destroyAllWindows();
    }
    //identify 
    private static void detectColor(Mat hsv, Mat frame, Scalar lower, Scalar upper, String colorName) {
        Mat mask = new Mat();
        Core.inRange(hsv, lower, upper, mask);

        Mat contours = new Mat();
        mask.copyTo(contours);
        //find counter
        java.util.List<MatOfPoint> contoursList = new java.util.ArrayList<>();
        Imgproc.findContours(contours, contoursList, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        // process every single counter
        for (MatOfPoint contour : contoursList) {
            MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(contour2f, approx, 0.02 * Imgproc.arcLength(contour2f, true), true);
            //labels color and if its a rectangle
            if (approx.total() == 4) {
                Rect rect = Imgproc.boundingRect(contour);
                Imgproc.rectangle(frame, rect, new Scalar(0, 255, 0), 2);
                Imgproc.putText(frame, colorName, new Point(rect.x, rect.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 255), 2);
            }
        }
    }
}



/*
 * Notes:
 *      Masks- are used to seperate wanted and unwanted data
 *      Contours- are used to outline boundaries of specific data, or in our case the individual blocks
 *      
 *      HSV- (Hue, Saturation, Value) 
 *      Hue: Represents the color type and is typically expressed as a degree from 0 to 360
 *      Saturation: Indicates the intensity or purity of the color
 *      Value: Represents the brightness of the color
 *      lower and upper parameters represent the minimum and maximum values of the color range you want to detect in the image.
 * 
 * ToDO:
 * https://docs.google.com/document/d/1hp2BnQT_cPSauXF5NvJvT3-YeD-ZNL12HzbJJQwpx1Y/edit?usp=sharing
 */
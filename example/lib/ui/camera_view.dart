import 'dart:developer';
import 'dart:io';
import 'package:image/image.dart' as img;
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'box_widget.dart';
import 'camera_view_singleton.dart';

/// [CameraView] sends each frame for inference
class CameraView extends StatefulWidget {
  /// Callback to pass results after inference to [HomeView]
  final Function(
          List<ResultObjectDetection> recognitions, Duration inferenceTime)
      resultsCallback;
  final Function(String classification, Duration inferenceTime)
      resultsCallbackClassification;

  /// Constructor
  const CameraView(this.resultsCallback, this.resultsCallbackClassification,
      {Key? key})
      : super(key: key);
  @override
  _CameraViewState createState() => _CameraViewState();
}

class _CameraViewState extends State<CameraView> with WidgetsBindingObserver {
  /// List of available cameras
  late List<CameraDescription> cameras;
  late File capturedImageFile;
  bool continueCapturing = true;
  int successfulDetectionCount = 0;

  /// Controller
  CameraController? cameraController;
  bool isCapturing = false;
  /// true when inference is ongoing
  bool predicting = false;
  bool captureImage = false;

  /// true when inference is ongoing
  bool predictingObjectDetection = false;

  ModelObjectDetection? _objectModel;
  ClassificationModel? _imageModel;

  bool classification = false;
  int _camFrameRotation = 0;
  String errorMessage = "";
  @override
  void initState() {
    super.initState();
    initStateAsync();
  }

  //load your model
  Future loadModel() async {
    String pathImageModel = "assets/models/model_classification.pt";
    //String pathCustomModel = "assets/models/custom_model.ptl";
    String pathObjectDetectionModel = "assets/models/best.torchscript";
    try {
      _imageModel = await PytorchLite.loadClassificationModel(
          pathImageModel, 224, 224, 1000,
          labelPath: "assets/labels/label_classification_imageNet.txt");
      //_customModel = await PytorchLite.loadCustomModel(pathCustomModel);
      _objectModel = await PytorchLite.loadObjectDetectionModel(
          pathObjectDetectionModel, 5, 640, 640,
          labelPath: "assets/labels/label.txt",
          objectDetectionModelType: ObjectDetectionModelType.yolov8);
    } catch (e) {
      if (e is PlatformException) {
        print("only supported for android, Error is $e");
      } else {
        print("Error is $e");
      }
    }
  }

  void initStateAsync() async {
    WidgetsBinding.instance.addObserver(this);
    await loadModel();

    // Camera initialization
    try {
      initializeCamera();
    } on CameraException catch (e) {
      switch (e.code) {
        case 'CameraAccessDenied':
          errorMessage = ('You have denied camera access.');
          break;
        case 'CameraAccessDeniedWithoutPrompt':
          // iOS only
          errorMessage = ('Please go to Settings app to enable camera access.');
          break;
        case 'CameraAccessRestricted':
          // iOS only
          errorMessage = ('Camera access is restricted.');
          break;
        case 'AudioAccessDenied':
          errorMessage = ('You have denied audio access.');
          break;
        case 'AudioAccessDeniedWithoutPrompt':
          // iOS only
          errorMessage = ('Please go to Settings app to enable audio access.');
          break;
        case 'AudioAccessRestricted':
          // iOS only
          errorMessage = ('Audio access is restricted.');
          break;
        default:
          errorMessage = (e.toString());
          break;
      }
      setState(() {});
    }
    // Initially predicting = false
    setState(() {
      predicting = false;
    });
  }

  /// Initializes the camera by setting [cameraController]
  void initializeCamera() async {
    cameras = await availableCameras();

    var idx =
        cameras.indexWhere((c) => c.lensDirection == CameraLensDirection.back);
    if (idx < 0) {
      log("No Back camera found - weird");
      return;
    }

    var desc = cameras[idx];
    _camFrameRotation = Platform.isAndroid ? desc.sensorOrientation : 0;
    // cameras[0] for rear-camera
    cameraController = CameraController(desc, ResolutionPreset.medium,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.yuv420
            : ImageFormatGroup.bgra8888,
        enableAudio: false);

    cameraController?.initialize().then((_) async {
      // Stream of image passed to [onLatestImageAvailable] callback
      await cameraController?.startImageStream(onLatestImageAvailable);

      /// previewSize is size of each image frame captured by controller
      ///
      /// 352x288 on iOS, 240p (320x240) on Android with ResolutionPreset.low
      Size? previewSize = cameraController?.value.previewSize;

      /// previewSize is size of raw input image to the model
      CameraViewSingleton.inputImageSize = previewSize!;

      // the display width of image on screen is
      // same as screenWidth while maintaining the aspectRatio
      Size screenSize = MediaQuery.of(context).size;
      CameraViewSingleton.screenSize = screenSize;
      CameraViewSingleton.ratio = cameraController!.value.aspectRatio;
    });
  }

  @override
  Widget build(BuildContext context) {
    // Return empty container while the camera is not initialized
    if (cameraController == null || !cameraController!.value.isInitialized) {
      return Container();
    }

    return CameraPreview(cameraController!);
    //return cameraController!.buildPreview();

    // return AspectRatio(
    //     // aspectRatio: cameraController.value.aspectRatio,
    //     child: CameraPreview(cameraController));
  }

  runClassification(CameraImage cameraImage) async {
    if (predicting) {
      return;
    }
    if (!mounted) {
      return;
    }

    setState(() {
      predicting = true;
    });
    if (_imageModel != null) {
      // Start the stopwatch
      Stopwatch stopwatch = Stopwatch()..start();

      String imageClassification = await _imageModel!
          .getCameraImagePrediction(cameraImage, _camFrameRotation);
      // Stop the stopwatch
      stopwatch.stop();
      // print("imageClassification $imageClassification");
      widget.resultsCallbackClassification(
          imageClassification, stopwatch.elapsed);
    }
    if (!mounted) {
      return;
    }

    setState(() {
      predicting = false;
    });
  }

  Future<void> takePicture(ResultObjectDetection result) async {
    if (cameraController == null || !cameraController!.value.isInitialized) {
      return;
    }

    if (isCapturing || !continueCapturing) {
      return;
    }

    setState(() {
      isCapturing = true;
    });

    try {
      final Directory extDir = await getTemporaryDirectory();
      final String dirPath = '${extDir.path}/pictures/';
      await Directory(dirPath).create(recursive: true);
      final String filePath =
          '$dirPath${DateTime.now().millisecondsSinceEpoch}.png';

      XFile pictureFile = await cameraController!.takePicture();
      await File(pictureFile.path).copy(filePath);

      capturedImageFile = File(filePath);

      print('Picture taken and saved at: $filePath');

      // Stop capturing after taking the picture
      setState(() {
        continueCapturing = false;
      });
      // Show the captured image on the screen
      await cropImage(filePath, result);
      // Perform image cropping

    } catch (e) {
      print('Error taking picture: $e');
    } finally {
      setState(() {
        isCapturing = false;
      });
    }
  }
  void _showCroppedImage(String croppedFilePath) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return Dialog(
          child: Container(
            width: double.infinity,
            height: double.infinity,
            decoration: BoxDecoration(
              image: DecorationImage(
                image: FileImage(File(croppedFilePath)),
                fit: BoxFit.contain,
              ),
            ),
          ),
        );
      },
    );
  }



  Future<void> cropImage(String filePath, ResultObjectDetection result) async {
    if (_objectModel != null) {
      // Load the captured image
      img.Image capturedImage = img.decodeImage(File(filePath).readAsBytesSync())!;

      // Use the provided result for cropping
      ResultObjectDetection firstResult = result;

      if (Platform.isIOS) {
        // Calculate cropping parameters for Android
        int left = (firstResult.rect.top * capturedImage.height).round();
        int top = (firstResult.rect.left * capturedImage.width).round();
        int right = ((firstResult.rect.top + firstResult.rect.height) * capturedImage.width).round();
        int bottom = ((firstResult.rect.left + firstResult.rect.width) * capturedImage.height).round();

        // Perform cropping
        img.Image croppedImage = img.copyCrop(capturedImage, x: left, y: top, width: right - left, height: bottom - top);

        // Save the cropped image
        String croppedFilePath = '${filePath}_cropped.png';
        File(croppedFilePath).writeAsBytesSync(img.encodePng(croppedImage));

        print("Path--$croppedFilePath");
        _showCroppedImage(croppedFilePath);
        print('Cropped image saved at: $croppedFilePath');

      } else {
        // Calculate cropping parameters for Android
        int left = (firstResult.rect.left * capturedImage.width).round();
        int top = (firstResult.rect.top * capturedImage.height).round();
        int right = ((firstResult.rect.left + firstResult.rect.width) * capturedImage.width).round();
        int bottom = ((firstResult.rect.top + firstResult.rect.height) * capturedImage.height).round();

        // Perform cropping
        img.Image croppedImage = img.copyCrop(capturedImage, x: left, y: top, width: right - left, height: bottom - top);

        // Save the cropped image
        String croppedFilePath = '${filePath}_cropped.png';
        File(croppedFilePath).writeAsBytesSync(img.encodePng(croppedImage));

        print("Path--$croppedFilePath");
        _showCroppedImage(croppedFilePath);
        print('Cropped image saved at: $croppedFilePath');
      }

    }
  }


  Future<void> runObjectDetection(CameraImage cameraImage) async {
    if (predictingObjectDetection) {
      return;
    }
    if (!mounted) {
      return;
    }

    setState(() {
      predictingObjectDetection = true;
    });
    if (_objectModel != null) {
      // Start the stopwatch
      Stopwatch stopwatch = Stopwatch()..start();

      List<ResultObjectDetection> objDetect =
          await _objectModel!.getCameraImagePrediction(
        cameraImage,
        _camFrameRotation,
        minimumScore: 0.5,
        iOUThreshold: 0.5,
      );
      print("--objDetect: $objDetect.");
      String? label= "CCCD_Chip_FrontSide";
      String? classname="";
      for (var result in objDetect) {
        classname = result.className;
        if (classname?.trim() == label && result.score > 0.88) {
          print(result.className);
          successfulDetectionCount++;
          if (successfulDetectionCount == 3) {
            takePicture(result);
            successfulDetectionCount = 0; // Đặt lại giá trị biến đếm sau khi chụp hình
          }
        }
      }

      // Stop the stopwatch
      stopwatch.stop();
      // print("data outputted $objDetect");
      widget.resultsCallback(objDetect, stopwatch.elapsed);
    }
    if (!mounted) {
      return;
    }
    setState(() {
      predictingObjectDetection = false;
    });
  }

  /// Callback to receive each frame [CameraImage] perform inference on it
  onLatestImageAvailable(CameraImage cameraImage) async {
    // Make sure we are still mounted, the background thread can return a response after we navigate away from this
    // screen but before bg thread is killed
    if (!mounted) {
      return;
    }

    // log("will start prediction");
    // log("Converted camera image");

    // runClassification(cameraImage);
    runObjectDetection(cameraImage);

    // log("done prediction camera image");
    // Make sure we are still mounted, the background thread can return a response after we navigate away from this
    // screen but before bg thread is killed
    if (!mounted) {
      return;
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) async {
    if (!mounted) {
      return;
    }

    switch (state) {
      case AppLifecycleState.paused:
      // Pause capturing when the app is paused
        continueCapturing = false;
        cameraController!.stopImageStream();
        break;
      case AppLifecycleState.resumed:
      // Resume capturing when the app is resumed
        continueCapturing = true;
        if (!cameraController!.value.isStreamingImages) {
          await cameraController!.startImageStream(onLatestImageAvailable);
        }
        break;
      default:
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance!.removeObserver(this);
    cameraController!.dispose();
    super.dispose();
  }
}


# Project Write-Up

## Explaining Custom Layers
Custom layers are layers that are not included into a list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

The list of known layers is different for each of supported frameworks. To see the layers supported by your framework, refer to the corresponding section https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html

Model Optimizer searches for each layer of the input model in the list of known layers before building the model's internal representation, optimizing the model, and producing the Intermediate Representation.


_The process behind converting custom layers involves...
the following steps_

    1. open terminal/CMD and activate the OpenVINO toolkit environment
    2. install prerequesite: pip install cogapp ..
    3. Run Model Extension Generator (tool for Model Optimizer): 
        This creates "code stubs" that will be edited in steps 6 and 8 with the custom algorithm.
    4. Edit C++ Code (produced by Model Extension Generator)
    5. Edit Python Scripts (produced by Model Extension Generator)
    6. Workaround for Linux
	Move a python custom layer script to the Model Optimizer operations directory:
	/opt/intel/openvino/deployment_tools/model_optimizer/mo/ops/
    7. Run the Model Optimizer
    8. Compile your C++ code.
    9. Test with Python and/or C++ sample apps.

HANDLING CUSTOM LAYER

To actually add custom layers, there are a few differences depending on the original model framework. In both TensorFlow and Caffe, the first option is to register the custom layers as extensions to the Model Optimizer.\
there are unsupported layers for certain hardware, that you may run into when working with the Inference Engine. In this case, there are sometimes extensions available that can add support. for example, the cpu extension for linux in `/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so`

_reasons for handling custom layers_
- a model with custom layer can not be converted successfully into IR with mo. the model optimizer return an error with custom layers name that should be implemented before conversion.
- If your topology contains layers that are not in the list of known layers for the device, the Inference Engine considers the layer to be unsupported and reports an error. handling custom layers will allow inference ingine to load the network with your model, perform some inference and return the results.


## Comparing Model Performance
Model Name: SSD MobileNet V2 COCO\
 My method(s) to compare models before and after conversion to Intermediate Representations
were...

    running the original (pre-conversion) model using tensorflow and post-conversion model using openvino toolkit

The difference between model accuracy pre- and post-conversion was:

    pre-conversion model: fails to track some people continuously, there is loose in accuracy
    post-conversion model: can not detect all the people if prob threshold with value >=5 is used

The size of the model pre- and post-conversion was:

    pre-conversion model : frozen_inference_graph.pb = 69.7 MB (69,688,296 bytes)
    post-conversion model: frozen_inference_graph.(bin/xml) = 67.4 MB (67,384,904 bytes)
    Difference: 2.30 MB

Inference Time

    the post-conversion model (IR) has less inference time than pre-conversion model: 149ms & 70ms avg
    also pre-trained model from open model zoo has less inference time than one from tensorflow: 45ms avg

The CPU Overhead of the model pre- and post-conversion was...

    pre conversion model: Around 60%/c
    post conversion model: Around 35%/c

compare the differences in network needs and costs of using cloud services as opposed to deploying at the edge...
   - Network communication can be expensive (bandwidth, power consumption, etc.): using cloud, connectivity, data migration, bandwidth, and latency features are pretty expensive. This inefficiency is remedied by edge, which has a significantly less bandwidth requirement and less latency.
   - cloud makes it difficult to process data gathered from the edge of the network quickly and effectively. edge enabled devices can gather and process data in real time and using local network connection only allowing them to respond faster and more effectively.
   - edge applications provide lower latency and reduce transmission costs.
   - there is also a cost of using cloud server, most of them you have to pay.

## Assess Model Use Cases

the people counter app play a good role in so many areas of part of our daily acivities including Retails store, shopping malls, public transportation, smart buildings etc. 

Some of the potential use cases of the people counter app are:
   - Retail Traffic Counters
   - Smart Office and smart buildings
   - Area monitoring
   - Security System

these use cases would be useful because:

- Shopping centers can use the people counters to measure the number of visitors in a given area. using it in more than one area will assist in measuring the areas where people tend to congregate. 
- People counting app can counts people per floors and identifies how many visitors are inside the building. 
- it can be used to count how many peoples visited a specific area or zone and the time they spend, by getting total count and duration that being used in this app. 
- also, it be used to specify a limit and get a message when the limit exceed the number of peoples needed in an area.
- for security, it can be used to restrict people access to a particular area, and get a message when people go there
- this will also help during this covid19 pandemic

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows:

    lighting : lighting plays a critical role in enabling the AI model to detect the object. for example low light can results in decreasing object detection rate, proper light is required for good result.

    accuracy: higher accuracy model require a lot of resources but low accuracy model can lead to getting false result, the end user should go for model with high accuracy (if he is ready) for better and accurate result.

    camera focal length: there is a limitation of how many meters the object on the camera will be out of focus. for monitoring wider place, camera with high focal length will be better, then for monitoring very narrow place, low focal length camera is enough. 

    image size: there is a models that need higher resolution image for input, these models can take more time than those that require lower resolution images to gives output. low quality image may loose it's quality during image preprocessing more espicially if the input size of model is greater than the the size of the resolution of given image. if end user has enough resources, high resolution image (with good quality) should be used for accurate result. 


## Model Research

In investigating potential people counter models, I tried each of the following three models:

- Model 1: 

  - Name : MobileNet SSD (caffee model)
  - Model Source : https://github.com/RamanHaivaronski/People-counter/tree/master/mobilenet_ssd
  - I converted the model to an Intermediate Representation with the following arguments... 

        --input_model --input_proto
  - command:

        python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model MobileNetSSD_deploy.caffemodel --input_proto MobileNetSSD_deploy.prototxt
  - The model was insufficient for the app because... the accuracy of model is not good enough
  - I tried to improve the model for the app during model conversion by... using mo.py with custom values for --mean_values data & --scale but still the problem is not solved


- Model 2: 
  - Name : faster_rcnn_inception_v2_coco
  - [Model Source] : http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments...

        --input_model
        --tensorflow_object_detection_api_pipeline_config
        --reverse_input_channels
        --tensorflow_use_custom_operations_config

  - command:
  
        python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model model/faster_rcnn_inception_v2_coco/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
  - The model was insufficient for the app due to... performance issues e.g slow inference time and hence is not suitable for this edge app.


- Model 3: 
   - Name : SSD MobileNet V2 COCO model
   - Model Source : http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz 
  - I converted the model to an Intermediate Representation with the following arguments...

          --input_model
          --tensorflow_object_detection_api_pipeline_config
          --reverse_input_channels
          --tensorflow_use_custom_operations_config
  - command : 
        
        python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  - The model was insufficient for the app because... The model lost some accuracy, it detects people but fails to continuously track them, and this result to false counting result. 
  - I tried to improve the model for the app by...  by reducing the probablity threshold, changing people counting algorithm, but still unable to detect two people in the video and still i am getting false result in counting.
  - the performance of this model is much better than the previous models.

## Conclusion
 I ended up using pre-trained model: Person-detection-retail-0013 model from OpenVINO model zoo which i found as suitable model for this application.

 ### command to download this pretrained model
    python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name person-detection-retail-0013 -o /home/workspace/

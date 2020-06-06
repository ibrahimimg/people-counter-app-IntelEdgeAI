#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.net_plugin = None
        self.infer_request_handle = None

    def load_model(self, model, device="CPU", cpu_extension=None):
        ### TODO: Load the model ###
        #Get the IR files
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # Initialize the plugin
        self.plugin = IECore()
        
        ### TODO: Add any necessary extensions ###
        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)
        
        #Load the Intermediate Representation model
        self.network = IENetwork(model=model_xml, weights=model_bin)
        
        ### TODO: Check for supported layers ###
        supported_layers = self.plugin.query_network(self.network, device)
        not_supported_layers = list(filter(lambda l: l not in supported_layers, map(lambda l: l, self.network.layers.keys() )))
        if len(not_supported_layers) != 0:
            print("Following layers are not supported: ")
            for i in range(len(not_supported_layers)):
                #print: id. layer name
                print (i+1, ". ",not_supported_layers[i])
            exit(1)

        #Load the model network
        self.net_plugin = self.plugin.load_network(self.network, device)
        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        ### TODO: Return the loaded inference plugin ###
        return self.plugin

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape

    def exec_net(self,r_id,image):
        ### parameter r_id: request id
        ### TODO: Start an asynchronous request ###
        self.infer_request_handle = self.net_plugin.start_async(request_id=r_id,inputs={self.input_blob: image})
        ### TODO: Return any necessary information ###
        return

    def wait(self,r_id):
        ### TODO: Wait for the request to be complete. ###
        status = self.net_plugin.requests[r_id].wait(-1)
        ### TODO: Return any necessary information ###
        return status

    def get_output(self,r_id,output=None):
        ### TODO: Extract and return the output results
        if not output:
            return self.net_plugin.requests[r_id].outputs[self.output_blob]
        return self.infer_request_handle.outputs[output]
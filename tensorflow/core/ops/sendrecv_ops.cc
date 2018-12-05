/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("_Send")
    .Input("tensor: T")
    .Attr("T: type")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .Attr("client_terminated: bool = false")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Sends the named tensor from send_device to recv_device.

tensor: The tensor to send.
tensor_name: The name of the tensor to send.
send_device: The name of the device sending the tensor.
send_device_incarnation: The current incarnation of send_device.
recv_device: The name of the device receiving the tensor.
client_terminated: If set to true, this indicates that the node was added
  to the graph as a result of a client-side feed or fetch of Tensor data,
  in which case the corresponding send or recv is expected to be managed
  locally by the caller.
)doc");

REGISTER_OP("_Recv")
    .Output("tensor: tensor_type")
    .Attr("tensor_type: type")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .Attr("client_terminated: bool = false")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Receives the named tensor from send_device on recv_device.

tensor: The tensor to receive.
tensor_name: The name of the tensor to receive.
send_device: The name of the device sending the tensor.
send_device_incarnation: The current incarnation of send_device.
recv_device: The name of the device receiving the tensor.
client_terminated: If set to true, this indicates that the node was added
  to the graph as a result of a client-side feed or fetch of Tensor data,
  in which case the corresponding send or recv is expected to be managed
  locally by the caller.
)doc");

REGISTER_OP("_HostSend")
    .Input("tensor: T")
    .Attr("T: type")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .Attr("client_terminated: bool = false")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Sends the named tensor from send_device to recv_device.

_HostSend requires its input on host memory whereas _Send requires its
input on device memory.

tensor: The tensor to send.
tensor_name: The name of the tensor to send.
send_device: The name of the device sending the tensor.
send_device_incarnation: The current incarnation of send_device.
recv_device: The name of the device receiving the tensor.
client_terminated: If set to true, this indicates that the node was added
  to the graph as a result of a client-side feed or fetch of Tensor data,
  in which case the corresponding send or recv is expected to be managed
  locally by the caller.
)doc");

REGISTER_OP("_HostRecv")
    .Output("tensor: tensor_type")
    .Attr("tensor_type: type")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .Attr("client_terminated: bool = false")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Receives the named tensor from send_device on recv_device.

_HostRecv requires its input on host memory whereas _Recv requires its
input on device memory.

tensor: The tensor to receive.
tensor_name: The name of the tensor to receive.
send_device: The name of the device sending the tensor.
send_device_incarnation: The current incarnation of send_device.
recv_device: The name of the device receiving the tensor.
client_terminated: If set to true, this indicates that the node was added
  to the graph as a result of a client-side feed or fetch of Tensor data,
  in which case the corresponding send or recv is expected to be managed
  locally by the caller.
)doc");

// New OP registration for {Send, Recv} nodes, which use SendTensor Service
REGISTER_OP("_stSend")
    .Input("tensor: T")
    .Attr("T: type")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .Attr("client_terminated: bool = false")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Sends the named tensor from send_device to recv_device.
(st stands for sendTensor)

Active push Tensor to ps instead of waiting for recv's request.
Used only when send_device is worker and recv_device is parameter server.
)doc");

REGISTER_OP("_stRecv")
    .Output("tensor: tensor_type")
    .Attr("tensor_type: type")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .Attr("client_terminated: bool = false")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Receives the named tensor from send_device on recv_device.
(st stands for sendTensor)

Passive waiting for request of _stSend on workers.
Used only when send_device is worker and recv_device is parameter server.
)doc");

REGISTER_OP("_stHostSend")
    .Input("tensor: T")
    .Attr("T: type")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .Attr("client_terminated: bool = false")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Sends the named tensor from send_device to recv_device.

_stHostSend requires its input on host memory whereas _stSend requires its
input on device memory.

Active push Tensor to ps instead of waiting for recv's request.
Used only when send_device is worker and recv_device is parameter server.
)doc");

REGISTER_OP("_stHostRecv")
    .Output("tensor: tensor_type")
    .Attr("tensor_type: type")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .Attr("client_terminated: bool = false")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Receives the named tensor from send_device on recv_device.

_stHostRecv requires its input on host memory whereas _stRecv requires its
input on device memory.

Passive waiting for request of _stSend on workers.
Used only when send_device is worker and recv_device is parameter server.
)doc");

}  // end namespace tensorflow

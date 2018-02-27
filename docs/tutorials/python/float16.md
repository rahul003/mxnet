# Using float16 on supported devices

In this tutorial we'll walk through how once can make use of float16 support which is now supported by the latest Volta range of NVidia GPUs.
The float16 data type also known as half precision reduces memory usage of the model, allowing the training and deployment of larger models.
Besides this, they would also reduce the time for transfer of data when compared to using float32 (single precision) or float64 (double precision).

## Prerequisites

Using float16 does not necessarily lead to faster performance, unless the underlying hardware has native support for the data type and
the workload is large enough to benefit from the optimizations. Examples of such hardware are the new Volta range of Graphics Processing Units by Nvidia.
For the rest of this tutorial we assume we are using such Nvidia GPUs.

- Volta or newer range of Nvidia GPUs
- Cuda 9 or higher
- CUDNN v7 or higher


## Using the Symbol API

MXNet's layers can generally work with any data type. Operators infer the datatype from input data and perform computation with that datatype.
So to enable training or inference in float16, all we need to do is add a Cast layer before the first operator.
In this tutorial we will build a model which trains imagenet data with a VGG-11 network.
You can see the complete example in [train_imagenet.py](example/image-classification/train_imagenet.py) and the network defined in [vgg.py](example/image-classification/symbols/vgg.py).

```python

def get_feature(internal_layer, layers, filters, batch_norm = False, **kwargs):
    for i, num in enumerate(layers):
        for j in range(num):
            internal_layer = mx.sym.Convolution(data = internal_layer, kernel=(3, 3), pad=(1, 1), num_filter=filters[i], name="conv%s_%s" %(i + 1, j + 1))
            if batch_norm:
                internal_layer = mx.symbol.BatchNorm(data=internal_layer, name="bn%s_%s" %(i + 1, j + 1))
            internal_layer = mx.sym.Activation(data=internal_layer, act_type="relu", name="relu%s_%s" %(i + 1, j + 1))
        internal_layer = mx.sym.Pooling(data=internal_layer, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool%s" %(i + 1))
    return internal_layer

def get_classifier(input_data, num_classes, **kwargs):
    flatten = mx.sym.Flatten(data=input_data, name="flatten")
    fc6 = mx.sym.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.sym.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.sym.Dropout(data=relu6, p=0.5, name="drop6")
    fc7 = mx.sym.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.sym.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.sym.Dropout(data=relu7, p=0.5, name="drop7")
    fc8 = mx.sym.FullyConnected(data=drop7, num_hidden=num_classes, name="fc8")
    return fc8

class SyntheticDataIter(DataIter):
    def __init__(self, num_classes, data_shape, max_iter, dtype):
        self.batch_size = data_shape[0]
        self.cur_iter = 0
        self.max_iter = max_iter
        self.dtype = dtype
        label = np.random.randint(0, num_classes, [self.batch_size,])
        data = np.random.uniform(-1, 1, data_shape)
        self.data = mx.nd.array(data, dtype=self.dtype, ctx=mx.Context('cpu_pinned', 0))
        self.label = mx.nd.array(label, dtype=self.dtype, ctx=mx.Context('cpu_pinned', 0))
    def __iter__(self):
        return self
    @property
    def provide_data(self):
        return [mx.io.DataDesc('data', self.data.shape, self.dtype)]
    @property
    def provide_label(self):
        return [mx.io.DataDesc('softmax_label', (self.batch_size,), self.dtype)]
    def next(self):
        self.cur_iter += 1
        if self.cur_iter <= self.max_iter:
            return DataBatch(data=(self.data,),
                             label=(self.label,),
                             pad=0,
                             index=None,
                             provide_data=self.provide_data,
                             provide_label=self.provide_label)
        else:
            raise StopIteration
    def __next__(self):
        return self.next()
    def reset(self):
        self.cur_iter = 0

def get_benchmark_data_iterators(batch_size):
    image_shape = (3,224,224)
    data_shape = (batch_size,) + image_shape
    train = SyntheticDataIter(args.num_classes, data_shape,
            args.num_examples / args.batch_size, np.float32)
    return (train, None)

def fit(batch_size, gpus=None):
    kv = mx.kvstore.create('local')
    # if you have gpus change this to 'device'

    # data iterators
    (train, val) = get_benchmark_data_iterators(batch_size)
    if args.test_io:

    # devices for training
    devs = mx.cpu()

    # create model
    model = mx.mod.Module(
        context=devs,
        symbol=network
    )

    optimizer_params = {
        'learning_rate': 0.01,
        'lr_scheduler': lr_scheduler,
        'multi_precision': True}

    # evaluation metrices
    eval_metrics = ['accuracy']

    # run
    model.fit(train,
              num_epoch=1,
              eval_data=val,
              kvstore='device',
              allow_missing=True)
```


<!-- INSERT SOURCE DOWNLOAD BUTTONS -->

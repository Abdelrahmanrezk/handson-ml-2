	Ve?????Ve?????!Ve?????	5e???@5e???@!5e???@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Ve??????'?X??A?O?????Y@?? kն?*	??x?&IY@2F
Iterator::Model??=$|???!???ɥ?N@)???߆??1?L7??LF@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat????1??!?=A?m5@)>?hɓ?1???K?3@:Preprocessing2S
Iterator::Model::ParallelMap?5v?ꭑ?!)??b?1@)?5v?ꭑ?1)??b?1@:Preprocessing2?
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice\?	??bu?!D?????@)\?	??bu?1D?????@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip?~2Ƈ٣?!lX_6Z*C@)?Fu:??t?1?F?<@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate?L??~ބ?!d??R&$@)????_Zt?1?r\%??@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor؜?gBc?!5???o?@)؜?gBc?15???o?@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap?,?뇈?!]8?Tn?'@)ʩ?ajK]?1???z?H??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2A7.9 % of the total step time sampled is spent on All Others time.>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?'?X???'?X??!?'?X??      ??!       "      ??!       *      ??!       2	?O??????O?????!?O?????:      ??!       B      ??!       J	@?? kն?@?? kն?!@?? kն?R      ??!       Z	@?? kն?@?? kն?!@?? kն?JCPU_ONLY
?	[??Ye??[??Ye??![??Ye??	??YZ?-@??YZ?-@!??YZ?-@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$[??Ye???J?h??A̴?++M??YDmFA???*	/?$?W@2F
Iterator::ModelL?K?1???!I??p?F@)A?M?Gş?1??D?S8@@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeatr??	???!?_O?5;@)X??Iؗ?1?y0??X8@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate(G?`Ɣ?!???p65@)MK??F??15?Dx??.@:Preprocessing2S
Iterator::Model::ParallelMap????7???!*q??8'@)????7???1*q??8'@:Preprocessing2X
!Iterator::Model::ParallelMap::Ziph???e??!?3?r?K@)B?????v?1U?M)!S@:Preprocessing2?
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??n/i?v?!???
@)??n/i?v?1???
@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor?8?? nf?!?0w???@)?8?? nf?1?0w???@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap??Po??!??˄T?6@)L?'??Z?1G??J:??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 14.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2B31.7 % of the total step time sampled is spent on All Others time.>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?J?h???J?h??!?J?h??      ??!       "      ??!       *      ??!       2	̴?++M??̴?++M??!̴?++M??:      ??!       B      ??!       J	DmFA???DmFA???!DmFA???R      ??!       Z	DmFA???DmFA???!DmFA???JCPU_ONLY2black"?
both?Your program is MODERATELY input-bound because 14.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendationN
nohigh"B31.7 % of the total step time sampled is spent on All Others time.:
Refer to the TF2 Profiler FAQ2"CPU: 
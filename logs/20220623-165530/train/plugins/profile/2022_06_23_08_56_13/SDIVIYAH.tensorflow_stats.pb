"?:
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
BHostIDLE"IDLE13333??AA3333??Aa?0?-???i?0?-????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(13333?Q?@93333?Q?@A3333?Q?@I3333?Q?@a0,N??z??i ??????Unknown?
iHostWriteSummary"WriteSummary(1??????@9??????@A??????@I??????@a???\?i?>?N????Unknown?
{HostReadVariableOp"Adam/Adam/update/ReadVariableOp(133333??@933333??@A33333??@I33333??@a?xQ?Y?iqz?7L????Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_2(133333;|@933333;|@A33333;|@I33333;|@aį??2?Q?i?nuQ????Unknown
?HostUnsortedSegmentSum"#Adam/Adam/update/UnsortedSegmentSum(1fffff?{@9fffff?{@Afffff?{@Ifffff?{@aXiG?:Q?iu#???????Unknown
?HostResourceGather")sequential_2/embedding_1/embedding_lookup(1????̔r@9????̔r@A????̔r@I????̔r@a^-:lG?i 24?r????Unknown
f	Host_Send"IteratorGetNext/_13(1?????,q@9?????,q@A?????,q@I?????,q@abG???TE?i?q??????Unknown
d
HostDataset"Iterator::Model(1fffff?o@9fffff?o@A33333?n@I33333?n@a?`???B?i??'!?????Unknown
kHostUnique"Adam/Adam/update/Unique(1??????l@9??????l@A??????l@I??????l@a<???p?A?i??U?????Unknown
?HostAssignVariableOp"#Adam/Adam/update/AssignVariableOp_1(1????̜g@9????̜g@A????̜g@I????̜g@a'??p?S=?i?̣7?????Unknown
?HostResourceScatterAdd"%Adam/Adam/update/ResourceScatterAdd_1(1??????[@9??????[@A??????[@I??????[@a
???K1?i8?t??????Unknown
eHost
LogicalAnd"
LogicalAnd(1      [@9      [@A      [@I      [@a?le?t?0?i?o9?????Unknown?
?HostResourceScatterAdd"#Adam/Adam/update/ResourceScatterAdd(1??????Z@9??????Z@A??????Z@I??????Z@a*???0?i8?{?????Unknown
?HostAssignSubVariableOp"$Adam/Adam/update/AssignSubVariableOp(1??????V@9??????V@A??????V@I??????V@a?sm?,?il?r^?????Unknown
gHostMul"Adam/Adam/update/mul_4(1fffff&S@9fffff&S@Afffff&S@Ifffff&S@a???X??'?i?H?M????Unknown
eHostMul"Adam/Adam/update/mul(1      S@9      S@A      S@I      S@a??g??'?iF????????Unknown
gHostMul"Adam/Adam/update/mul_2(1?????LN@9?????LN@A?????LN@I?????LN@aI?t??"?i?YY?????Unknown
gHostMul"Adam/Adam/update/mul_3(1??????M@9??????M@A??????M@I??????M@a?j?A??"?i?t?'????Unknown
HostAssignVariableOp"!Adam/Adam/update/AssignVariableOp(1     @L@9     @L@A     @L@I     @L@a?P.?!?i?y??7????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1??????J@9??????J@A      H@I      H@a?ݗ	??iX?S&????Unknown
gHostMul"Adam/Adam/update/mul_1(1?????G@9?????G@A?????G@I?????G@a?^?"ް?i???????Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_3(1      F@9      F@A      F@I      F@aw?u?S?i?x?r?????Unknown
?Host_Send"-sequential_2/embedding_1/embedding_lookup/_25(1     @E@9     @E@A     @E@I     @E@a?!)s?d?i?	??????Unknown
gHostMul"Adam/Adam/update/mul_5(133333sD@933333sD@A33333sD@I33333sD@a?"?PFf?i??;ʄ????Unknown
gHostSqrt"Adam/Adam/update/Sqrt(1ffffffD@9ffffffD@AffffffD@IffffffD@a???n`V?i?}O????Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_1(1??????@@9??????@@A??????@@I??????@@a?'?????il?4??????Unknown
mHostRealDiv"Adam/Adam/update/truediv(1fffff?@@9fffff?@@Afffff?@@Ifffff?@@a"(-??i?!nZ?????Unknown
?Host_Recv"Cgradient_tape/sequential_2/embedding_1/embedding_lookup/Reshape/_34(1ffffff<@9ffffff<@Affffff<@Iffffff<@a??????i4>?r)????Unknown
{Host_Send"(Adam/Adam/update/AssignSubVariableOp/_36(1ffffff;@9ffffff;@Affffff;@Iffffff;@a\l?iG???????Unknown
g HostAddV2"Adam/Adam/update/add(1?????L:@9?????L:@A?????L:@I?????L:@a'?RV+U?i?``<4????Unknown
?!HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      ?@9      ?@A333333:@I333333:@a9m+tEE?i7?f?????Unknown
^"HostGatherV2"GatherV2(1??????-@9??????-@A??????-@I??????-@a??a}?a?i?????????Unknown
f#Host_Send"IteratorGetNext/_11(1      -@9      -@A      -@I      -@a7kv0j?i??j?G????Unknown
p$Host_Recv"Adam/Cast_4/ReadVariableOp/_4(1ffffff+@9ffffff+@Affffff+@Iffffff+@a\l?i????????Unknown
?%HostVariableShape"Egradient_tape/sequential_2/embedding_1/embedding_lookup/VariableShape(1??????*@9??????*@A??????*@I??????*@a????? ?i??=??????Unknown
s&HostDataset"Iterator::Model::ParallelMapV2(1333333%@9333333%@A333333%@I333333%@a????T?>i?C????Unknown
?'Host	_HostSend"Fcategorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2/_17(1??????$@9??????$@A??????$@I??????$@a/?????>i???6????Unknown
k(Host_Recv"Adam/ReadVariableOp_1/_2(1??????$@9??????$@A??????$@I??????$@ay?*????>i??j????Unknown
p)Host_Recv"Adam/Cast_7/ReadVariableOp/_8(1333333 @9333333 @A333333 @I333333 @a???9??>iCy?X?????Unknown
p*Host_Recv"Adam/Cast_6/ReadVariableOp/_6(1      @9      @A      @I      @a???e??>iA?x??????Unknown
`+HostGatherV2"
GatherV2_1(1??????@9??????@A??????@I??????@at?,f??>iM??h?????Unknown
?,Host_Recv"Egradient_tape/sequential_2/embedding_1/embedding_lookup/Reshape_1/_28(1ffffff@9ffffff@Affffff@Iffffff@a?߯?K??>i?u*;?????Unknown
l-HostIteratorGetNext"IteratorGetNext(1333333@9333333@A333333@I333333@a????T?>i???????Unknown
?.HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1??????@9??????@A??????@I??????@a/?????>iǆxe&????Unknown
t/Host_Recv"!sequential_2/embedding_1/Cast/_24(1??????@9??????@A??????@I??????@a/?????>i?;@????Unknown
x0HostDataset"#Iterator::Model::ParallelMapV2::Zip(1??????V@9??????V@Affffff@Iffffff@a???n`V?>iuh?Y????Unknown
?1HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1333333@9333333@A333333@I333333@az??:???>i??;jq????Unknown
[2HostSub"
Adam/sub_7(1ffffff@9ffffff@Affffff@Iffffff@a6ꛎ???>i?>8K?????Unknown
]3HostCast"Adam/Cast_5(1??????@9??????@A??????@I??????@at?,f??>i?T뱓????Unknown
[4HostPow"
Adam/Pow_2(1      @9      @A      @I      @aU?S]1??>iI??????Unknown
]5HostAddV2"
Adam/add_1(1      @9      @A      @I      @aU?S]1??>i????????Unknown
?6HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1??????K@9??????K@A333333@I333333@az??:???>icO?u?????Unknown
x7HostStridedSlice"Adam/Adam/update/strided_slice(1?????? @9?????? @A?????? @I?????? @a???Ӹ??>i%?b??????Unknown
c8HostRealDiv"Adam/truediv_1(1       @9       @A       @I       @a??Z??>i???????Unknown
[9HostPow"
Adam/Pow_3(1333333??9333333??A333333??I333333??a???I@??>i?60F?????Unknown
[:HostSub"
Adam/sub_6(1333333??9333333??A333333??I333333??a???I@??>ia[P??????Unknown
?;Host_Send"Hgradient_tape/sequential_2/embedding_1/embedding_lookup/VariableShape/_9(1333333??9333333??A333333??I333333??a???I@??>i;?p*?????Unknown
]<HostSqrt"Adam/Sqrt_1(1????????9????????A????????I????????a/?????>i-`ԟ?????Unknown
[=HostMul"
Adam/mul_1(1333333??9333333??A333333??I333333??az??:???>i?.	??????Unknown
[>HostSub"
Adam/sub_5(1????????9????????A????????I????????a??????>ic??????Unknown
a?HostIdentity"Identity(1????????9????????A????????I????????a\?'l???>im????????Unknown?
[@HostSub"
Adam/sub_4(1ffffff??9ffffff??Affffff??Iffffff??a?߯?Kһ>i     ???Unknown*?9
uHostFlushSummaryWriter"FlushSummaryWriter(13333?Q?@93333?Q?@A3333?Q?@I3333?Q?@aN?w@??iN?w@???Unknown?
iHostWriteSummary"WriteSummary(1??????@9??????@A??????@I??????@a???cק?i??}?????Unknown?
{HostReadVariableOp"Adam/Adam/update/ReadVariableOp(133333??@933333??@A33333??@I33333??@a??zm????i????a????Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_2(133333;|@933333;|@A33333;|@I33333;|@aو)Y?A??i???On????Unknown
?HostUnsortedSegmentSum"#Adam/Adam/update/UnsortedSegmentSum(1fffff?{@9fffff?{@Afffff?{@Ifffff?{@a????6???i??#p????Unknown
?HostResourceGather")sequential_2/embedding_1/embedding_lookup(1????̔r@9????̔r@A????̔r@I????̔r@a???GxA??i?b?{Z???Unknown
fHost_Send"IteratorGetNext/_13(1?????,q@9?????,q@A?????,q@I?????,q@a?8?f̑?i;ګ??????Unknown
dHostDataset"Iterator::Model(1fffff?o@9fffff?o@A33333?n@I33333?n@a_???R???i?܆K?g???Unknown
k	HostUnique"Adam/Adam/update/Unique(1??????l@9??????l@A??????l@I??????l@a?J????i?շ?????Unknown
?
HostAssignVariableOp"#Adam/Adam/update/AssignVariableOp_1(1????̜g@9????̜g@A????̜g@I????̜g@a
ȼD9x??i;?蜘A???Unknown
?HostResourceScatterAdd"%Adam/Adam/update/ResourceScatterAdd_1(1??????[@9??????[@A??????[@I??????[@aW??f?|?if?xjQ{???Unknown
eHost
LogicalAnd"
LogicalAnd(1      [@9      [@A      [@I      [@a? q??{?ih??8G????Unknown?
?HostResourceScatterAdd"#Adam/Adam/update/ResourceScatterAdd(1??????Z@9??????Z@A??????Z@I??????Z@a??'??{?i?[p?????Unknown
?HostAssignSubVariableOp"$Adam/Adam/update/AssignSubVariableOp(1??????V@9??????V@A??????V@I??????V@a??E??w?i?}\?????Unknown
gHostMul"Adam/Adam/update/mul_4(1fffff&S@9fffff&S@Afffff&S@Ifffff&S@a?<|?Z?s?iBv'Y<B???Unknown
eHostMul"Adam/Adam/update/mul(1      S@9      S@A      S@I      S@aR???s?iYV3??i???Unknown
gHostMul"Adam/Adam/update/mul_2(1?????LN@9?????LN@A?????LN@I?????LN@aj??[^fo?i??)F????Unknown
gHostMul"Adam/Adam/update/mul_3(1??????M@9??????M@A??????M@I??????M@a??????n?i??s?'????Unknown
HostAssignVariableOp"!Adam/Adam/update/AssignVariableOp(1     @L@9     @L@A     @L@I     @L@a????Fm?iA?4gn????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1??????J@9??????J@A      H@I      H@a?ǹM?h?i	d?mM????Unknown
gHostMul"Adam/Adam/update/mul_1(1?????G@9?????G@A?????G@I?????G@ap??B?g?i?f`?=????Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_3(1      F@9      F@A      F@I      F@a?L?qp?f?i&Q? 
???Unknown
?Host_Send"-sequential_2/embedding_1/embedding_lookup/_25(1     @E@9     @E@A     @E@I     @E@a[~|?xf?i???#???Unknown
gHostMul"Adam/Adam/update/mul_5(133333sD@933333sD@A33333sD@I33333sD@aM?M<1e?i??_?@8???Unknown
gHostSqrt"Adam/Adam/update/Sqrt(1ffffffD@9ffffffD@AffffffD@IffffffD@a??ݎ?#e?iۡ??dM???Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_1(1??????@@9??????@@A??????@@I??????@@a
o?.va?iJ?}??^???Unknown
mHostRealDiv"Adam/Adam/update/truediv(1fffff?@@9fffff?@@Afffff?@@Ifffff?@@a?⸒Aa?i-up???Unknown
?Host_Recv"Cgradient_tape/sequential_2/embedding_1/embedding_lookup/Reshape/_34(1ffffff<@9ffffff<@Affffff<@Iffffff<@a@??On]?i?D?~???Unknown
{Host_Send"(Adam/Adam/update/AssignSubVariableOp/_36(1ffffff;@9ffffff;@Affffff;@Iffffff;@a?4e\?i&???????Unknown
gHostAddV2"Adam/Adam/update/add(1?????L:@9?????L:@A?????L:@I?????L:@aÕ?2A[?i???_?????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      ?@9      ?@A333333:@I333333:@a???%?&[?iY?|?9????Unknown
^ HostGatherV2"GatherV2(1??????-@9??????-@A??????-@I??????-@a@!e詬N?i?????????Unknown
f!Host_Send"IteratorGetNext/_11(1      -@9      -@A      -@I      -@a@|@?|N?i??3?h????Unknown
p"Host_Recv"Adam/Cast_4/ReadVariableOp/_4(1ffffff+@9ffffff+@Affffff+@Iffffff+@a?4eL?iƖw??????Unknown
?#HostVariableShape"Egradient_tape/sequential_2/embedding_1/embedding_lookup/VariableShape(1??????*@9??????*@A??????*@I??????*@a?t??K?i?Z~?r????Unknown
s$HostDataset"Iterator::Model::ParallelMapV2(1333333%@9333333%@A333333%@I333333%@aFd?4?E?i?s??????Unknown
?%Host	_HostSend"Fcategorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2/_17(1??????$@9??????$@A??????$@I??????$@a????E?i?[@?T????Unknown
k&Host_Recv"Adam/ReadVariableOp_1/_2(1??????$@9??????$@A??????$@I??????$@aGv??YE?i?+#˪????Unknown
p'Host_Recv"Adam/Cast_7/ReadVariableOp/_8(1333333 @9333333 @A333333 @I333333 @a?f?ڽ?@?i#??:?????Unknown
p(Host_Recv"Adam/Cast_6/ReadVariableOp/_6(1      @9      @A      @I      @a?9(????i*??????Unknown
`)HostGatherV2"
GatherV2_1(1??????@9??????@A??????@I??????@aD???B?9?i)??{?????Unknown
?*Host_Recv"Egradient_tape/sequential_2/embedding_1/embedding_lookup/Reshape_1/_28(1ffffff@9ffffff@Affffff@Iffffff@aEe?j?67?i?E?M?????Unknown
l+HostIteratorGetNext"IteratorGetNext(1333333@9333333@A333333@I333333@aFd?4?5?iYR;T?????Unknown
?,HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1??????@9??????@A??????@I??????@a????5?iyFN????Unknown
t-Host_Recv"!sequential_2/embedding_1/Cast/_24(1??????@9??????@A??????@I??????@a????5?i?:???????Unknown
x.HostDataset"#Iterator::Model::ParallelMapV2::Zip(1??????V@9??????V@Affffff@Iffffff@a??ݎ?#5?iV?X?????Unknown
?/HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1333333@9333333@A333333@I333333@a򟔤??3?i???!????Unknown
[0HostSub"
Adam/sub_7(1ffffff@9ffffff@Affffff@Iffffff@a?R????/?i?G!????Unknown
]1HostCast"Adam/Cast_5(1??????@9??????@A??????@I??????@aD???B?)?i?;EO?????Unknown
[2HostPow"
Adam/Pow_2(1      @9      @A      @I      @aG??ڹ$?iL????????Unknown
]3HostAddV2"
Adam/add_1(1      @9      @A      @I      @aG??ڹ$?i????K????Unknown
?4HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1??????K@9??????K@A333333@I333333@a򟔤??#?iCH???????Unknown
x5HostStridedSlice"Adam/Adam/update/strided_slice(1?????? @9?????? @A?????? @I?????? @a???h!?idH/s?????Unknown
c6HostRealDiv"Adam/truediv_1(1       @9       @A       @I       @a??{ޮ? ?i"0??????Unknown
[7HostPow"
Adam/Pow_3(1333333??9333333??A333333??I333333??aA???/?i???=?????Unknown
[8HostSub"
Adam/sub_6(1333333??9333333??A333333??I333333??aA???/?iJm~?l????Unknown
?9Host_Send"Hgradient_tape/sequential_2/embedding_1/embedding_lookup/VariableShape/_9(1333333??9333333??A333333??I333333??aA???/?i?/=N????Unknown
]:HostSqrt"Adam/Sqrt_1(1????????9????????A????????I????????a?????i?H???????Unknown
[;HostMul"
Adam/mul_1(1333333??9333333??A333333??I333333??a򟔤???i?m?ڙ????Unknown
[<HostSub"
Adam/sub_5(1????????9????????A????????I????????aI=??&=?i?y?+????Unknown
a=HostIdentity"Identity(1????????9????????A????????I????????a????m??iIU?%?????Unknown?
[>HostSub"
Adam/sub_4(1ffffff??9ffffff??Affffff??Iffffff??aEe?j?6?i?????????Unknown
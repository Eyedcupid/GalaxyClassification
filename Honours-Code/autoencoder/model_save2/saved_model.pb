ϲ 
� �
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
E
AssignAddVariableOp
resource
value"dtype"
dtypetype�
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

9
DivNoNan
x"T
y"T
z"T"
Ttype:

2
,
Exp
x"T
y"T"
Ttype:

2
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
.
Log1p
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
7
Square
x"T
y"T"
Ttype:
2	
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8��
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   �
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
Adam/conv2d_transpose_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_3/bias/v
�
2Adam/conv2d_transpose_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_3/bias/v*
_output_shapes
:*
dtype0
�
 Adam/conv2d_transpose_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_3/kernel/v
�
4Adam/conv2d_transpose_3/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_3/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/conv2d_transpose_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_2/bias/v
�
2Adam/conv2d_transpose_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_2/bias/v*
_output_shapes
:*
dtype0
�
 Adam/conv2d_transpose_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_2/kernel/v
�
4Adam/conv2d_transpose_2/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_2/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/conv2d_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_1/bias/v
�
2Adam/conv2d_transpose_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_1/bias/v*
_output_shapes
:*
dtype0
�
 Adam/conv2d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/conv2d_transpose_1/kernel/v
�
4Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_1/kernel/v*&
_output_shapes
: *
dtype0
�
Adam/conv2d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/conv2d_transpose/bias/v
�
0Adam/conv2d_transpose/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/bias/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  */
shared_name Adam/conv2d_transpose/kernel/v
�
2Adam/conv2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/kernel/v*&
_output_shapes
:  *
dtype0
�
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameAdam/dense_2/bias/v
y
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes

:��*
dtype0
�
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
@��*&
shared_nameAdam/dense_2/kernel/v
�
)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v* 
_output_shapes
:
@��*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:@*
dtype0
�
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��@*&
shared_nameAdam/dense_1/kernel/v
�
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v* 
_output_shapes
:
��@*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:@*
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��@*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
��@*
dtype0
�
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_2/kernel/v
�
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
: *
dtype0
�
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/v
�
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v
�
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/conv2d_transpose_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_3/bias/m
�
2Adam/conv2d_transpose_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_3/bias/m*
_output_shapes
:*
dtype0
�
 Adam/conv2d_transpose_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_3/kernel/m
�
4Adam/conv2d_transpose_3/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_3/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/conv2d_transpose_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_2/bias/m
�
2Adam/conv2d_transpose_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_2/bias/m*
_output_shapes
:*
dtype0
�
 Adam/conv2d_transpose_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_2/kernel/m
�
4Adam/conv2d_transpose_2/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_2/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/conv2d_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_1/bias/m
�
2Adam/conv2d_transpose_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_1/bias/m*
_output_shapes
:*
dtype0
�
 Adam/conv2d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/conv2d_transpose_1/kernel/m
�
4Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_1/kernel/m*&
_output_shapes
: *
dtype0
�
Adam/conv2d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/conv2d_transpose/bias/m
�
0Adam/conv2d_transpose/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/bias/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  */
shared_name Adam/conv2d_transpose/kernel/m
�
2Adam/conv2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/kernel/m*&
_output_shapes
:  *
dtype0
�
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameAdam/dense_2/bias/m
y
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes

:��*
dtype0
�
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
@��*&
shared_nameAdam/dense_2/kernel/m
�
)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m* 
_output_shapes
:
@��*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��@*&
shared_nameAdam/dense_1/kernel/m
�
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m* 
_output_shapes
:
��@*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��@*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
��@*
dtype0
�
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_2/kernel/m
�
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
: *
dtype0
�
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/m
�
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m
�
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
x
add_metric_2/countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameadd_metric_2/count
q
&add_metric_2/count/Read/ReadVariableOpReadVariableOpadd_metric_2/count*
_output_shapes
: *
dtype0
x
add_metric_2/totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameadd_metric_2/total
q
&add_metric_2/total/Read/ReadVariableOpReadVariableOpadd_metric_2/total*
_output_shapes
: *
dtype0
x
add_metric_1/countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameadd_metric_1/count
q
&add_metric_1/count/Read/ReadVariableOpReadVariableOpadd_metric_1/count*
_output_shapes
: *
dtype0
x
add_metric_1/totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameadd_metric_1/total
q
&add_metric_1/total/Read/ReadVariableOpReadVariableOpadd_metric_1/total*
_output_shapes
: *
dtype0
t
add_metric/countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameadd_metric/count
m
$add_metric/count/Read/ReadVariableOpReadVariableOpadd_metric/count*
_output_shapes
: *
dtype0
t
add_metric/totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameadd_metric/total
m
$add_metric/total/Read/ReadVariableOpReadVariableOpadd_metric/total*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
�
conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_3/bias

+conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/bias*
_output_shapes
:*
dtype0
�
conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_3/kernel
�
-conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/kernel*&
_output_shapes
:*
dtype0
�
conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_2/bias

+conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/bias*
_output_shapes
:*
dtype0
�
conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_2/kernel
�
-conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/kernel*&
_output_shapes
:*
dtype0
�
conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_1/bias

+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes
:*
dtype0
�
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_transpose_1/kernel
�
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*&
_output_shapes
: *
dtype0
�
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameconv2d_transpose/bias
{
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes
: *
dtype0
�
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameconv2d_transpose/kernel
�
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*&
_output_shapes
:  *
dtype0
r
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*
shared_namedense_2/bias
k
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes

:��*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
@��*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
@��*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��@*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
��@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��@*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
��@*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
: *
dtype0
�
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
�
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
�
serving_default_input_1Placeholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense_1/kerneldense_1/biasdense/kernel
dense/biasdense_2/kerneldense_2/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasConst_1add_metric/totaladd_metric/countConstadd_metric_1/totaladd_metric_1/countadd_metric_2/totaladd_metric_2/count*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *-
f(R&
$__inference_signature_wrapper_240453

NoOpNoOp
ޮ
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-1
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_default_save_signature
$	optimizer
%loss
&
signatures*
* 
�
layer-0
'layer_with_weights-0
'layer-1
(layer_with_weights-1
(layer-2
)layer_with_weights-2
)layer-3
*layer-4
+layer_with_weights-3
+layer-5
,layer_with_weights-4
,layer-6
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses*

3	keras_api* 

4	keras_api* 

5	keras_api* 

6	keras_api* 

7	keras_api* 

8	keras_api* 

9	keras_api* 

:	keras_api* 
�
;layer-0
<layer_with_weights-0
<layer-1
=layer-2
>layer_with_weights-1
>layer-3
?layer_with_weights-2
?layer-4
@layer_with_weights-3
@layer-5
Alayer_with_weights-4
Alayer-6
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses*

H	keras_api* 

I	keras_api* 

J	keras_api* 

K	keras_api* 

L	keras_api* 

M	keras_api* 

N	keras_api* 

O	keras_api* 

P	keras_api* 

Q	keras_api* 

R	keras_api* 
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses* 

Y	keras_api* 
�
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses*

`	keras_api* 
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses*
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses*
�
m0
n1
o2
p3
q4
r5
s6
t7
u8
v9
w10
x11
y12
z13
{14
|15
}16
~17
18
�19*
�
m0
n1
o2
p3
q4
r5
s6
t7
u8
v9
w10
x11
y12
z13
{14
|15
}16
~17
18
�19*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
#_default_save_signature
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
$
�
capture_20
�
capture_23* 
�
	�iter
�beta_1
�beta_2

�decay
�learning_ratemm�nm�om�pm�qm�rm�sm�tm�um�vm�wm�xm�ym�zm�{m�|m�}m�~m�m�	�m�mv�nv�ov�pv�qv�rv�sv�tv�uv�vv�wv�xv�yv�zv�{v�|v�}v�~v�v�	�v�*
* 

�serving_default* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

mkernel
nbias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

okernel
pbias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

qkernel
rbias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

skernel
tbias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

ukernel
vbias*
J
m0
n1
o2
p3
q4
r5
s6
t7
u8
v9*
J
m0
n1
o2
p3
q4
r5
s6
t7
u8
v9*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

wkernel
xbias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

ykernel
zbias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

{kernel
|bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

}kernel
~bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
	�bias
!�_jit_compiled_convolution_op*
K
w0
x1
y2
z3
{4
|5
}6
~7
8
�9*
K
w0
x1
y2
z3
{4
|5
}6
~7
8
�9*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
MG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
dense/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEconv2d_transpose/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_1/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_1/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_2/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_2/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_3/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27*
$
�0
�1
�2
�3*
* 
* 
$
�
capture_20
�
capture_23* 
$
�
capture_20
�
capture_23* 
$
�
capture_20
�
capture_23* 
$
�
capture_20
�
capture_23* 
$
�
capture_20
�
capture_23* 
$
�
capture_20
�
capture_23* 
$
�
capture_20
�
capture_23* 
$
�
capture_20
�
capture_23* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
$
�
capture_20
�
capture_23* 

m0
n1*

m0
n1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

o0
p1*

o0
p1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

q0
r1*

q0
r1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

s0
t1*

s0
t1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

u0
v1*

u0
v1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
5
0
'1
(2
)3
*4
+5
,6*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

w0
x1*

w0
x1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

y0
z1*

y0
z1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

{0
|1*

{0
|1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

}0
~1*

}0
~1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

0
�1*

0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
5
;0
<1
=2
>3
?4
@5
A6*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0*
* 

�reconstruction_loss*
* 
* 
* 
* 

�0*
* 

�kl_loss*
* 
* 
* 
* 

�0*
* 

�	elbo_loss*
* 
* 
<
�	variables
�	keras_api

�total

�count*
<
�	variables
�	keras_api

�total

�count*
<
�	variables
�	keras_api

�total

�count*
<
�	variables
�	keras_api

�total

�count*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
^X
VARIABLE_VALUEadd_metric/total4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEadd_metric/count4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
`Z
VARIABLE_VALUEadd_metric_1/total4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEadd_metric_1/count4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
`Z
VARIABLE_VALUEadd_metric_2/total4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEadd_metric_2/count4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv2d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEAdam/dense/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_1/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_1/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_2/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_2/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/conv2d_transpose/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_1/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_1/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_2/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_2/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_3/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_3/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv2d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEAdam/dense/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_1/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_1/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_2/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_2/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/conv2d_transpose/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_1/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_1/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_2/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_2/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_3/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_3/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp-conv2d_transpose_2/kernel/Read/ReadVariableOp+conv2d_transpose_2/bias/Read/ReadVariableOp-conv2d_transpose_3/kernel/Read/ReadVariableOp+conv2d_transpose_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp$add_metric/total/Read/ReadVariableOp$add_metric/count/Read/ReadVariableOp&add_metric_1/total/Read/ReadVariableOp&add_metric_1/count/Read/ReadVariableOp&add_metric_2/total/Read/ReadVariableOp&add_metric_2/count/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp2Adam/conv2d_transpose/kernel/m/Read/ReadVariableOp0Adam/conv2d_transpose/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_1/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_2/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_2/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_3/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_3/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp2Adam/conv2d_transpose/kernel/v/Read/ReadVariableOp0Adam/conv2d_transpose/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_1/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_2/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_2/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_3/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_3/bias/v/Read/ReadVariableOpConst_2*V
TinO
M2K	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *(
f#R!
__inference__traced_save_242068
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d_transpose_3/kernelconv2d_transpose_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountadd_metric/totaladd_metric/countadd_metric_1/totaladd_metric_1/countadd_metric_2/totaladd_metric_2/countAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/conv2d_transpose/kernel/mAdam/conv2d_transpose/bias/m Adam/conv2d_transpose_1/kernel/mAdam/conv2d_transpose_1/bias/m Adam/conv2d_transpose_2/kernel/mAdam/conv2d_transpose_2/bias/m Adam/conv2d_transpose_3/kernel/mAdam/conv2d_transpose_3/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/conv2d_transpose/kernel/vAdam/conv2d_transpose/bias/v Adam/conv2d_transpose_1/kernel/vAdam/conv2d_transpose_1/bias/v Adam/conv2d_transpose_2/kernel/vAdam/conv2d_transpose_2/bias/v Adam/conv2d_transpose_3/kernel/vAdam/conv2d_transpose_3/bias/v*U
TinN
L2J*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *+
f&R$
"__inference__traced_restore_242297��
�

�
C__inference_dense_2_layer_call_and_return_conditional_losses_239294

inputs2
matmul_readvariableop_resource:
@��/
biasadd_readvariableop_resource:
��
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
@��*
dtype0k
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������t
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:��*
dtype0x
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������R
ReluReluBiasAdd:output:0*
T0*)
_output_shapes
:�����������c
IdentityIdentityRelu:activations:0^NoOp*
T0*)
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
A__inference_dense_layer_call_and_return_conditional_losses_238833

inputs2
matmul_readvariableop_resource:
��@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
D
(__inference_flatten_layer_call_fn_241569

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_238805b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_238776

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������88i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������88w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������pp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������pp
 
_user_specified_nameinputs
�i
�	
C__inference_model_2_layer_call_and_return_conditional_losses_240384
input_1&
model_240263:
model_240265:&
model_240267:
model_240269:&
model_240271: 
model_240273:  
model_240275:
��@
model_240277:@ 
model_240279:
��@
model_240281:@"
model_1_240304:
@��
model_1_240306:
��(
model_1_240308:  
model_1_240310: (
model_1_240312: 
model_1_240314:(
model_1_240316:
model_1_240318:(
model_1_240320:
model_1_240322:
unknown
add_metric_240357: 
add_metric_240359: 
	unknown_0
add_metric_1_240370: 
add_metric_1_240372: 
add_metric_2_240377: 
add_metric_2_240379: 
identity

identity_1��"add_metric/StatefulPartitionedCall�$add_metric_1/StatefulPartitionedCall�$add_metric_2/StatefulPartitionedCall�model/StatefulPartitionedCall�model_1/StatefulPartitionedCall�
model/StatefulPartitionedCallStatefulPartitionedCallinput_1model_240263model_240265model_240267model_240269model_240271model_240273model_240275model_240277model_240279model_240281*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_238982p
tf.compat.v1.shape_1/ShapeShape&model/StatefulPartitionedCall:output:1*
T0*
_output_shapes
:n
tf.compat.v1.shape/ShapeShape&model/StatefulPartitionedCall:output:1*
T0*
_output_shapes
:v
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&tf.__operators__.getitem/strided_sliceStridedSlice!tf.compat.v1.shape/Shape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(tf.__operators__.getitem_1/strided_sliceStridedSlice#tf.compat.v1.shape_1/Shape:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
tf.math.exp/ExpExp&model/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:���������@�
$tf.random.normal/random_normal/shapePack/tf.__operators__.getitem/strided_slice:output:01tf.__operators__.getitem_1/strided_slice:output:0*
N*
T0*
_output_shapes
:h
#tf.random.normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%tf.random.normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
3tf.random.normal/random_normal/RandomStandardNormalRandomStandardNormal-tf.random.normal/random_normal/shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0�
"tf.random.normal/random_normal/mulMul<tf.random.normal/random_normal/RandomStandardNormal:output:0.tf.random.normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:���������@�
tf.random.normal/random_normalAddV2&tf.random.normal/random_normal/mul:z:0,tf.random.normal/random_normal/mean:output:0*
T0*'
_output_shapes
:���������@�
tf.math.multiply/MulMultf.math.exp/Exp:y:0"tf.random.normal/random_normal:z:0*
T0*'
_output_shapes
:���������@�
tf.__operators__.add/AddV2AddV2&model/StatefulPartitionedCall:output:0tf.math.multiply/Mul:z:0*
T0*'
_output_shapes
:���������@�
model_1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0model_1_240304model_1_240306model_1_240308model_1_240310model_1_240312model_1_240314model_1_240316model_1_240318model_1_240320model_1_240322*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_239433o
*tf.keras.backend.binary_crossentropy/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���3o
*tf.keras.backend.binary_crossentropy/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
(tf.keras.backend.binary_crossentropy/subSub3tf.keras.backend.binary_crossentropy/sub/x:output:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*
_output_shapes
: �
:tf.keras.backend.binary_crossentropy/clip_by_value/MinimumMinimum(model_1/StatefulPartitionedCall:output:0,tf.keras.backend.binary_crossentropy/sub:z:0*
T0*1
_output_shapes
:������������
2tf.keras.backend.binary_crossentropy/clip_by_valueMaximum>tf.keras.backend.binary_crossentropy/clip_by_value/Minimum:z:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*1
_output_shapes
:�����������o
*tf.keras.backend.binary_crossentropy/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
(tf.keras.backend.binary_crossentropy/addAddV26tf.keras.backend.binary_crossentropy/clip_by_value:z:03tf.keras.backend.binary_crossentropy/add/y:output:0*
T0*1
_output_shapes
:������������
(tf.keras.backend.binary_crossentropy/LogLog,tf.keras.backend.binary_crossentropy/add:z:0*
T0*1
_output_shapes
:������������
(tf.keras.backend.binary_crossentropy/mulMulinput_1,tf.keras.backend.binary_crossentropy/Log:y:0*
T0*1
_output_shapes
:�����������q
,tf.keras.backend.binary_crossentropy/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
*tf.keras.backend.binary_crossentropy/sub_1Sub5tf.keras.backend.binary_crossentropy/sub_1/x:output:0input_1*
T0*1
_output_shapes
:�����������q
,tf.keras.backend.binary_crossentropy/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
*tf.keras.backend.binary_crossentropy/sub_2Sub5tf.keras.backend.binary_crossentropy/sub_2/x:output:06tf.keras.backend.binary_crossentropy/clip_by_value:z:0*
T0*1
_output_shapes
:�����������q
,tf.keras.backend.binary_crossentropy/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
*tf.keras.backend.binary_crossentropy/add_1AddV2.tf.keras.backend.binary_crossentropy/sub_2:z:05tf.keras.backend.binary_crossentropy/add_1/y:output:0*
T0*1
_output_shapes
:������������
*tf.keras.backend.binary_crossentropy/Log_1Log.tf.keras.backend.binary_crossentropy/add_1:z:0*
T0*1
_output_shapes
:������������
*tf.keras.backend.binary_crossentropy/mul_1Mul.tf.keras.backend.binary_crossentropy/sub_1:z:0.tf.keras.backend.binary_crossentropy/Log_1:y:0*
T0*1
_output_shapes
:������������
*tf.keras.backend.binary_crossentropy/add_2AddV2,tf.keras.backend.binary_crossentropy/mul:z:0.tf.keras.backend.binary_crossentropy/mul_1:z:0*
T0*1
_output_shapes
:������������
(tf.keras.backend.binary_crossentropy/NegNeg.tf.keras.backend.binary_crossentropy/add_2:z:0*
T0*1
_output_shapes
:�����������r
tf.math.exp_1/ExpExp&model/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:���������@�
tf.__operators__.add_1/AddV2AddV2unknown&model/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:���������@y
tf.math.square/SquareSquare&model/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@}
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         �
tf.math.reduce_sum/SumSum,tf.keras.backend.binary_crossentropy/Neg:y:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:����������
tf.math.subtract/SubSub tf.__operators__.add_1/AddV2:z:0tf.math.square/Square:y:0*
T0*'
_output_shapes
:���������@�
tf.math.subtract_1/SubSubtf.math.subtract/Sub:z:0tf.math.exp_1/Exp:y:0*
T0*'
_output_shapes
:���������@e
tf.math.reduce_mean_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_mean_1/MeanMeantf.math.reduce_sum/Sum:output:0$tf.math.reduce_mean_1/Const:output:0*
T0*
_output_shapes
: u
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
tf.math.reduce_sum_1/SumSumtf.math.subtract_1/Sub:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:����������
"add_metric/StatefulPartitionedCallStatefulPartitionedCall#tf.math.reduce_mean_1/Mean:output:0add_metric_240357add_metric_240359*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_add_metric_layer_call_and_return_conditional_losses_239659y
tf.math.multiply_1/MulMul	unknown_0!tf.math.reduce_sum_1/Sum:output:0*
T0*#
_output_shapes
:����������
tf.__operators__.add_2/AddV2AddV2tf.math.reduce_sum/Sum:output:0tf.math.multiply_1/Mul:z:0*
T0*#
_output_shapes
:���������e
tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_mean_2/MeanMeantf.math.multiply_1/Mul:z:0$tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: c
tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_mean/MeanMean tf.__operators__.add_2/AddV2:z:0"tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: �
$add_metric_1/StatefulPartitionedCallStatefulPartitionedCall#tf.math.reduce_mean_2/Mean:output:0add_metric_1_240370add_metric_1_240372*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_add_metric_1_layer_call_and_return_conditional_losses_239690�
add_loss/PartitionedCallPartitionedCall!tf.math.reduce_mean/Mean:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_add_loss_layer_call_and_return_conditional_losses_239701�
$add_metric_2/StatefulPartitionedCallStatefulPartitionedCall!tf.math.reduce_mean/Mean:output:0add_metric_2_240377add_metric_2_240379*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_add_metric_2_layer_call_and_return_conditional_losses_239721�
IdentityIdentity(model_1/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������a

Identity_1Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: �
NoOpNoOp#^add_metric/StatefulPartitionedCall%^add_metric_1/StatefulPartitionedCall%^add_metric_2/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"add_metric/StatefulPartitionedCall"add_metric/StatefulPartitionedCall2L
$add_metric_1/StatefulPartitionedCall$add_metric_1/StatefulPartitionedCall2L
$add_metric_2/StatefulPartitionedCall$add_metric_2/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: 
��
�	
C__inference_model_1_layer_call_and_return_conditional_losses_241415

inputs:
&dense_2_matmul_readvariableop_resource:
@��7
'dense_2_biasadd_readvariableop_resource:
��S
9conv2d_transpose_conv2d_transpose_readvariableop_resource:  >
0conv2d_transpose_biasadd_readvariableop_resource: U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_1_biasadd_readvariableop_resource:U
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_2_biasadd_readvariableop_resource:U
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_3_biasadd_readvariableop_resource:
identity��'conv2d_transpose/BiasAdd/ReadVariableOp�0conv2d_transpose/conv2d_transpose/ReadVariableOp�)conv2d_transpose_1/BiasAdd/ReadVariableOp�2conv2d_transpose_1/conv2d_transpose/ReadVariableOp�)conv2d_transpose_2/BiasAdd/ReadVariableOp�2conv2d_transpose_2/conv2d_transpose/ReadVariableOp�)conv2d_transpose_3/BiasAdd/ReadVariableOp�2conv2d_transpose_3/conv2d_transpose/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
@��*
dtype0{
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:������������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes

:��*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������b
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*)
_output_shapes
:�����������W
reshape/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : �
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape/ReshapeReshapedense_2/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� ^
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :8Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :8Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0�
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:���������88 *
paddingSAME*
strides
�
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88 z
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:���������88 k
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :p\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :p\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp~
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������ppm
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�]
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�\
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0�
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
conv2d_transpose_2/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*1
_output_shapes
:�����������m
conv2d_transpose_3/ShapeShape%conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�]
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0�
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_2/Relu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
conv2d_transpose_3/SigmoidSigmoid#conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:�����������w
IdentityIdentityconv2d_transpose_3/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�i
�	
C__inference_model_2_layer_call_and_return_conditional_losses_240014

inputs&
model_239893:
model_239895:&
model_239897:
model_239899:&
model_239901: 
model_239903:  
model_239905:
��@
model_239907:@ 
model_239909:
��@
model_239911:@"
model_1_239934:
@��
model_1_239936:
��(
model_1_239938:  
model_1_239940: (
model_1_239942: 
model_1_239944:(
model_1_239946:
model_1_239948:(
model_1_239950:
model_1_239952:
unknown
add_metric_239987: 
add_metric_239989: 
	unknown_0
add_metric_1_240000: 
add_metric_1_240002: 
add_metric_2_240007: 
add_metric_2_240009: 
identity

identity_1��"add_metric/StatefulPartitionedCall�$add_metric_1/StatefulPartitionedCall�$add_metric_2/StatefulPartitionedCall�model/StatefulPartitionedCall�model_1/StatefulPartitionedCall�
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_239893model_239895model_239897model_239899model_239901model_239903model_239905model_239907model_239909model_239911*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_238982p
tf.compat.v1.shape_1/ShapeShape&model/StatefulPartitionedCall:output:1*
T0*
_output_shapes
:n
tf.compat.v1.shape/ShapeShape&model/StatefulPartitionedCall:output:1*
T0*
_output_shapes
:v
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&tf.__operators__.getitem/strided_sliceStridedSlice!tf.compat.v1.shape/Shape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(tf.__operators__.getitem_1/strided_sliceStridedSlice#tf.compat.v1.shape_1/Shape:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
tf.math.exp/ExpExp&model/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:���������@�
$tf.random.normal/random_normal/shapePack/tf.__operators__.getitem/strided_slice:output:01tf.__operators__.getitem_1/strided_slice:output:0*
N*
T0*
_output_shapes
:h
#tf.random.normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%tf.random.normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
3tf.random.normal/random_normal/RandomStandardNormalRandomStandardNormal-tf.random.normal/random_normal/shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0�
"tf.random.normal/random_normal/mulMul<tf.random.normal/random_normal/RandomStandardNormal:output:0.tf.random.normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:���������@�
tf.random.normal/random_normalAddV2&tf.random.normal/random_normal/mul:z:0,tf.random.normal/random_normal/mean:output:0*
T0*'
_output_shapes
:���������@�
tf.math.multiply/MulMultf.math.exp/Exp:y:0"tf.random.normal/random_normal:z:0*
T0*'
_output_shapes
:���������@�
tf.__operators__.add/AddV2AddV2&model/StatefulPartitionedCall:output:0tf.math.multiply/Mul:z:0*
T0*'
_output_shapes
:���������@�
model_1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0model_1_239934model_1_239936model_1_239938model_1_239940model_1_239942model_1_239944model_1_239946model_1_239948model_1_239950model_1_239952*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_239433o
*tf.keras.backend.binary_crossentropy/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���3o
*tf.keras.backend.binary_crossentropy/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
(tf.keras.backend.binary_crossentropy/subSub3tf.keras.backend.binary_crossentropy/sub/x:output:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*
_output_shapes
: �
:tf.keras.backend.binary_crossentropy/clip_by_value/MinimumMinimum(model_1/StatefulPartitionedCall:output:0,tf.keras.backend.binary_crossentropy/sub:z:0*
T0*1
_output_shapes
:������������
2tf.keras.backend.binary_crossentropy/clip_by_valueMaximum>tf.keras.backend.binary_crossentropy/clip_by_value/Minimum:z:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*1
_output_shapes
:�����������o
*tf.keras.backend.binary_crossentropy/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
(tf.keras.backend.binary_crossentropy/addAddV26tf.keras.backend.binary_crossentropy/clip_by_value:z:03tf.keras.backend.binary_crossentropy/add/y:output:0*
T0*1
_output_shapes
:������������
(tf.keras.backend.binary_crossentropy/LogLog,tf.keras.backend.binary_crossentropy/add:z:0*
T0*1
_output_shapes
:������������
(tf.keras.backend.binary_crossentropy/mulMulinputs,tf.keras.backend.binary_crossentropy/Log:y:0*
T0*1
_output_shapes
:�����������q
,tf.keras.backend.binary_crossentropy/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
*tf.keras.backend.binary_crossentropy/sub_1Sub5tf.keras.backend.binary_crossentropy/sub_1/x:output:0inputs*
T0*1
_output_shapes
:�����������q
,tf.keras.backend.binary_crossentropy/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
*tf.keras.backend.binary_crossentropy/sub_2Sub5tf.keras.backend.binary_crossentropy/sub_2/x:output:06tf.keras.backend.binary_crossentropy/clip_by_value:z:0*
T0*1
_output_shapes
:�����������q
,tf.keras.backend.binary_crossentropy/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
*tf.keras.backend.binary_crossentropy/add_1AddV2.tf.keras.backend.binary_crossentropy/sub_2:z:05tf.keras.backend.binary_crossentropy/add_1/y:output:0*
T0*1
_output_shapes
:������������
*tf.keras.backend.binary_crossentropy/Log_1Log.tf.keras.backend.binary_crossentropy/add_1:z:0*
T0*1
_output_shapes
:������������
*tf.keras.backend.binary_crossentropy/mul_1Mul.tf.keras.backend.binary_crossentropy/sub_1:z:0.tf.keras.backend.binary_crossentropy/Log_1:y:0*
T0*1
_output_shapes
:������������
*tf.keras.backend.binary_crossentropy/add_2AddV2,tf.keras.backend.binary_crossentropy/mul:z:0.tf.keras.backend.binary_crossentropy/mul_1:z:0*
T0*1
_output_shapes
:������������
(tf.keras.backend.binary_crossentropy/NegNeg.tf.keras.backend.binary_crossentropy/add_2:z:0*
T0*1
_output_shapes
:�����������r
tf.math.exp_1/ExpExp&model/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:���������@�
tf.__operators__.add_1/AddV2AddV2unknown&model/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:���������@y
tf.math.square/SquareSquare&model/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@}
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         �
tf.math.reduce_sum/SumSum,tf.keras.backend.binary_crossentropy/Neg:y:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:����������
tf.math.subtract/SubSub tf.__operators__.add_1/AddV2:z:0tf.math.square/Square:y:0*
T0*'
_output_shapes
:���������@�
tf.math.subtract_1/SubSubtf.math.subtract/Sub:z:0tf.math.exp_1/Exp:y:0*
T0*'
_output_shapes
:���������@e
tf.math.reduce_mean_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_mean_1/MeanMeantf.math.reduce_sum/Sum:output:0$tf.math.reduce_mean_1/Const:output:0*
T0*
_output_shapes
: u
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
tf.math.reduce_sum_1/SumSumtf.math.subtract_1/Sub:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:����������
"add_metric/StatefulPartitionedCallStatefulPartitionedCall#tf.math.reduce_mean_1/Mean:output:0add_metric_239987add_metric_239989*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_add_metric_layer_call_and_return_conditional_losses_239659y
tf.math.multiply_1/MulMul	unknown_0!tf.math.reduce_sum_1/Sum:output:0*
T0*#
_output_shapes
:����������
tf.__operators__.add_2/AddV2AddV2tf.math.reduce_sum/Sum:output:0tf.math.multiply_1/Mul:z:0*
T0*#
_output_shapes
:���������e
tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_mean_2/MeanMeantf.math.multiply_1/Mul:z:0$tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: c
tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_mean/MeanMean tf.__operators__.add_2/AddV2:z:0"tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: �
$add_metric_1/StatefulPartitionedCallStatefulPartitionedCall#tf.math.reduce_mean_2/Mean:output:0add_metric_1_240000add_metric_1_240002*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_add_metric_1_layer_call_and_return_conditional_losses_239690�
add_loss/PartitionedCallPartitionedCall!tf.math.reduce_mean/Mean:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_add_loss_layer_call_and_return_conditional_losses_239701�
$add_metric_2/StatefulPartitionedCallStatefulPartitionedCall!tf.math.reduce_mean/Mean:output:0add_metric_2_240007add_metric_2_240009*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_add_metric_2_layer_call_and_return_conditional_losses_239721�
IdentityIdentity(model_1/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������a

Identity_1Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: �
NoOpNoOp#^add_metric/StatefulPartitionedCall%^add_metric_1/StatefulPartitionedCall%^add_metric_2/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"add_metric/StatefulPartitionedCall"add_metric/StatefulPartitionedCall2L
$add_metric_1/StatefulPartitionedCall$add_metric_1/StatefulPartitionedCall2L
$add_metric_2/StatefulPartitionedCall$add_metric_2/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
B__inference_conv2d_layer_call_and_return_conditional_losses_238759

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ppX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������ppi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������ppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_238866
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
��@
	unknown_6:@
	unknown_7:
��@
	unknown_8:@
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_238841o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�

�
(__inference_model_1_layer_call_fn_239360
input_2
unknown:
@��
	unknown_0:
��#
	unknown_1:  
	unknown_2: #
	unknown_3: 
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_239337y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������@
!
_user_specified_name	input_2
�

�
(__inference_model_1_layer_call_fn_241188

inputs
unknown:
@��
	unknown_0:
��#
	unknown_1:  
	unknown_2: #
	unknown_3: 
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_239337y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�!
�
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_239224

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�	
�
C__inference_dense_1_layer_call_and_return_conditional_losses_241613

inputs2
matmul_readvariableop_resource:
��@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�!
�
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_241695

inputsB
(conv2d_transpose_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�"
�
C__inference_model_1_layer_call_and_return_conditional_losses_239337

inputs"
dense_2_239295:
@��
dense_2_239297:
��1
conv2d_transpose_239316:  %
conv2d_transpose_239318: 3
conv2d_transpose_1_239321: '
conv2d_transpose_1_239323:3
conv2d_transpose_2_239326:'
conv2d_transpose_2_239328:3
conv2d_transpose_3_239331:'
conv2d_transpose_3_239333:
identity��(conv2d_transpose/StatefulPartitionedCall�*conv2d_transpose_1/StatefulPartitionedCall�*conv2d_transpose_2/StatefulPartitionedCall�*conv2d_transpose_3/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_239295dense_2_239297*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_239294�
reshape/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_239314�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_239316conv2d_transpose_239318*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_239134�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_239321conv2d_transpose_1_239323*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_239179�
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_239326conv2d_transpose_2_239328*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_239224�
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_239331conv2d_transpose_3_239333*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_239269�
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_241083

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
��@
	unknown_6:@
	unknown_7:
��@
	unknown_8:@
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_238982o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_240453
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
��@
	unknown_6:@
	unknown_7:
��@
	unknown_8:@
	unknown_9:
@��

unknown_10:
��$

unknown_11:  

unknown_12: $

unknown_13: 

unknown_14:$

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19

unknown_20: 

unknown_21: 

unknown_22

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� **
f%R#
!__inference__wrapped_model_238741y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: 
��
�
C__inference_model_2_layer_call_and_return_conditional_losses_240803

inputsE
+model_conv2d_conv2d_readvariableop_resource::
,model_conv2d_biasadd_readvariableop_resource:G
-model_conv2d_1_conv2d_readvariableop_resource:<
.model_conv2d_1_biasadd_readvariableop_resource:G
-model_conv2d_2_conv2d_readvariableop_resource: <
.model_conv2d_2_biasadd_readvariableop_resource: @
,model_dense_1_matmul_readvariableop_resource:
��@;
-model_dense_1_biasadd_readvariableop_resource:@>
*model_dense_matmul_readvariableop_resource:
��@9
+model_dense_biasadd_readvariableop_resource:@B
.model_1_dense_2_matmul_readvariableop_resource:
@��?
/model_1_dense_2_biasadd_readvariableop_resource:
��[
Amodel_1_conv2d_transpose_conv2d_transpose_readvariableop_resource:  F
8model_1_conv2d_transpose_biasadd_readvariableop_resource: ]
Cmodel_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource: H
:model_1_conv2d_transpose_1_biasadd_readvariableop_resource:]
Cmodel_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:H
:model_1_conv2d_transpose_2_biasadd_readvariableop_resource:]
Cmodel_1_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:H
:model_1_conv2d_transpose_3_biasadd_readvariableop_resource:
unknown1
'add_metric_assignaddvariableop_resource: 3
)add_metric_assignaddvariableop_1_resource: 
	unknown_03
)add_metric_1_assignaddvariableop_resource: 5
+add_metric_1_assignaddvariableop_1_resource: 3
)add_metric_2_assignaddvariableop_resource: 5
+add_metric_2_assignaddvariableop_1_resource: 
identity

identity_1��add_metric/AssignAddVariableOp� add_metric/AssignAddVariableOp_1�$add_metric/div_no_nan/ReadVariableOp�&add_metric/div_no_nan/ReadVariableOp_1� add_metric_1/AssignAddVariableOp�"add_metric_1/AssignAddVariableOp_1�&add_metric_1/div_no_nan/ReadVariableOp�(add_metric_1/div_no_nan/ReadVariableOp_1� add_metric_2/AssignAddVariableOp�"add_metric_2/AssignAddVariableOp_1�&add_metric_2/div_no_nan/ReadVariableOp�(add_metric_2/div_no_nan/ReadVariableOp_1�#model/conv2d/BiasAdd/ReadVariableOp�"model/conv2d/Conv2D/ReadVariableOp�%model/conv2d_1/BiasAdd/ReadVariableOp�$model/conv2d_1/Conv2D/ReadVariableOp�%model/conv2d_2/BiasAdd/ReadVariableOp�$model/conv2d_2/Conv2D/ReadVariableOp�"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�/model_1/conv2d_transpose/BiasAdd/ReadVariableOp�8model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp�1model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp�:model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp�1model_1/conv2d_transpose_2/BiasAdd/ReadVariableOp�:model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp�1model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp�:model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp�&model_1/dense_2/BiasAdd/ReadVariableOp�%model_1/dense_2/MatMul/ReadVariableOp�
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
model/conv2d/Conv2DConv2Dinputs*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ppr
model/conv2d/ReluRelumodel/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp�
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
model/conv2d_1/Conv2DConv2Dmodel/conv2d/Relu:activations:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88v
model/conv2d_1/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������88�
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
model/conv2d_2/Conv2DConv2D!model/conv2d_1/Relu:activations:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� v
model/conv2d_2/ReluRelumodel/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:��������� d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� b  �
model/flatten/ReshapeReshape!model/conv2d_2/Relu:activations:0model/flatten/Const:output:0*
T0*)
_output_shapes
:������������
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��@*
dtype0�
model/dense_1/MatMulMatMulmodel/flatten/Reshape:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource* 
_output_shapes
:
��@*
dtype0�
model/dense/MatMulMatMulmodel/flatten/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@h
tf.compat.v1.shape_1/ShapeShapemodel/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:f
tf.compat.v1.shape/ShapeShapemodel/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:v
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&tf.__operators__.getitem/strided_sliceStridedSlice!tf.compat.v1.shape/Shape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(tf.__operators__.getitem_1/strided_sliceStridedSlice#tf.compat.v1.shape_1/Shape:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
tf.math.exp/ExpExpmodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
$tf.random.normal/random_normal/shapePack/tf.__operators__.getitem/strided_slice:output:01tf.__operators__.getitem_1/strided_slice:output:0*
N*
T0*
_output_shapes
:h
#tf.random.normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%tf.random.normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
3tf.random.normal/random_normal/RandomStandardNormalRandomStandardNormal-tf.random.normal/random_normal/shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0�
"tf.random.normal/random_normal/mulMul<tf.random.normal/random_normal/RandomStandardNormal:output:0.tf.random.normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:���������@�
tf.random.normal/random_normalAddV2&tf.random.normal/random_normal/mul:z:0,tf.random.normal/random_normal/mean:output:0*
T0*'
_output_shapes
:���������@�
tf.math.multiply/MulMultf.math.exp/Exp:y:0"tf.random.normal/random_normal:z:0*
T0*'
_output_shapes
:���������@�
tf.__operators__.add/AddV2AddV2model/dense/BiasAdd:output:0tf.math.multiply/Mul:z:0*
T0*'
_output_shapes
:���������@�
%model_1/dense_2/MatMul/ReadVariableOpReadVariableOp.model_1_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
@��*
dtype0�
model_1/dense_2/MatMulMatMultf.__operators__.add/AddV2:z:0-model_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:������������
&model_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_2_biasadd_readvariableop_resource*
_output_shapes

:��*
dtype0�
model_1/dense_2/BiasAddBiasAdd model_1/dense_2/MatMul:product:0.model_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������r
model_1/dense_2/ReluRelu model_1/dense_2/BiasAdd:output:0*
T0*)
_output_shapes
:�����������g
model_1/reshape/ShapeShape"model_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:m
#model_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model_1/reshape/strided_sliceStridedSlicemodel_1/reshape/Shape:output:0,model_1/reshape/strided_slice/stack:output:0.model_1/reshape/strided_slice/stack_1:output:0.model_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :a
model_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :a
model_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : �
model_1/reshape/Reshape/shapePack&model_1/reshape/strided_slice:output:0(model_1/reshape/Reshape/shape/1:output:0(model_1/reshape/Reshape/shape/2:output:0(model_1/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
model_1/reshape/ReshapeReshape"model_1/dense_2/Relu:activations:0&model_1/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� n
model_1/conv2d_transpose/ShapeShape model_1/reshape/Reshape:output:0*
T0*
_output_shapes
:v
,model_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&model_1/conv2d_transpose/strided_sliceStridedSlice'model_1/conv2d_transpose/Shape:output:05model_1/conv2d_transpose/strided_slice/stack:output:07model_1/conv2d_transpose/strided_slice/stack_1:output:07model_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 model_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :8b
 model_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :8b
 model_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
model_1/conv2d_transpose/stackPack/model_1/conv2d_transpose/strided_slice:output:0)model_1/conv2d_transpose/stack/1:output:0)model_1/conv2d_transpose/stack/2:output:0)model_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:x
.model_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(model_1/conv2d_transpose/strided_slice_1StridedSlice'model_1/conv2d_transpose/stack:output:07model_1/conv2d_transpose/strided_slice_1/stack:output:09model_1/conv2d_transpose/strided_slice_1/stack_1:output:09model_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
8model_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0�
)model_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput'model_1/conv2d_transpose/stack:output:0@model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0 model_1/reshape/Reshape:output:0*
T0*/
_output_shapes
:���������88 *
paddingSAME*
strides
�
/model_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp8model_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
 model_1/conv2d_transpose/BiasAddBiasAdd2model_1/conv2d_transpose/conv2d_transpose:output:07model_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88 �
model_1/conv2d_transpose/ReluRelu)model_1/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:���������88 {
 model_1/conv2d_transpose_1/ShapeShape+model_1/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:x
.model_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(model_1/conv2d_transpose_1/strided_sliceStridedSlice)model_1/conv2d_transpose_1/Shape:output:07model_1/conv2d_transpose_1/strided_slice/stack:output:09model_1/conv2d_transpose_1/strided_slice/stack_1:output:09model_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"model_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :pd
"model_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :pd
"model_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
 model_1/conv2d_transpose_1/stackPack1model_1/conv2d_transpose_1/strided_slice:output:0+model_1/conv2d_transpose_1/stack/1:output:0+model_1/conv2d_transpose_1/stack/2:output:0+model_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:z
0model_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2model_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2model_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*model_1/conv2d_transpose_1/strided_slice_1StridedSlice)model_1/conv2d_transpose_1/stack:output:09model_1/conv2d_transpose_1/strided_slice_1/stack:output:0;model_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
:model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
+model_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_1/stack:output:0Bmodel_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0+model_1/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
1model_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
"model_1/conv2d_transpose_1/BiasAddBiasAdd4model_1/conv2d_transpose_1/conv2d_transpose:output:09model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp�
model_1/conv2d_transpose_1/ReluRelu+model_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp}
 model_1/conv2d_transpose_2/ShapeShape-model_1/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:x
.model_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(model_1/conv2d_transpose_2/strided_sliceStridedSlice)model_1/conv2d_transpose_2/Shape:output:07model_1/conv2d_transpose_2/strided_slice/stack:output:09model_1/conv2d_transpose_2/strided_slice/stack_1:output:09model_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"model_1/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�e
"model_1/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�d
"model_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
 model_1/conv2d_transpose_2/stackPack1model_1/conv2d_transpose_2/strided_slice:output:0+model_1/conv2d_transpose_2/stack/1:output:0+model_1/conv2d_transpose_2/stack/2:output:0+model_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:z
0model_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2model_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2model_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*model_1/conv2d_transpose_2/strided_slice_1StridedSlice)model_1/conv2d_transpose_2/stack:output:09model_1/conv2d_transpose_2/strided_slice_1/stack:output:0;model_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
:model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0�
+model_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_2/stack:output:0Bmodel_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0-model_1/conv2d_transpose_1/Relu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
1model_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
"model_1/conv2d_transpose_2/BiasAddBiasAdd4model_1/conv2d_transpose_2/conv2d_transpose:output:09model_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
model_1/conv2d_transpose_2/ReluRelu+model_1/conv2d_transpose_2/BiasAdd:output:0*
T0*1
_output_shapes
:�����������}
 model_1/conv2d_transpose_3/ShapeShape-model_1/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:x
.model_1/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_1/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_1/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(model_1/conv2d_transpose_3/strided_sliceStridedSlice)model_1/conv2d_transpose_3/Shape:output:07model_1/conv2d_transpose_3/strided_slice/stack:output:09model_1/conv2d_transpose_3/strided_slice/stack_1:output:09model_1/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"model_1/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�e
"model_1/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�d
"model_1/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
 model_1/conv2d_transpose_3/stackPack1model_1/conv2d_transpose_3/strided_slice:output:0+model_1/conv2d_transpose_3/stack/1:output:0+model_1/conv2d_transpose_3/stack/2:output:0+model_1/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:z
0model_1/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2model_1/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2model_1/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*model_1/conv2d_transpose_3/strided_slice_1StridedSlice)model_1/conv2d_transpose_3/stack:output:09model_1/conv2d_transpose_3/strided_slice_1/stack:output:0;model_1/conv2d_transpose_3/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
:model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0�
+model_1/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_3/stack:output:0Bmodel_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0-model_1/conv2d_transpose_2/Relu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
1model_1/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
"model_1/conv2d_transpose_3/BiasAddBiasAdd4model_1/conv2d_transpose_3/conv2d_transpose:output:09model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
"model_1/conv2d_transpose_3/SigmoidSigmoid+model_1/conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:������������
=tf.keras.backend.binary_crossentropy/logistic_loss/zeros_like	ZerosLike+model_1/conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:������������
?tf.keras.backend.binary_crossentropy/logistic_loss/GreaterEqualGreaterEqual+model_1/conv2d_transpose_3/BiasAdd:output:0Atf.keras.backend.binary_crossentropy/logistic_loss/zeros_like:y:0*
T0*1
_output_shapes
:������������
9tf.keras.backend.binary_crossentropy/logistic_loss/SelectSelectCtf.keras.backend.binary_crossentropy/logistic_loss/GreaterEqual:z:0+model_1/conv2d_transpose_3/BiasAdd:output:0Atf.keras.backend.binary_crossentropy/logistic_loss/zeros_like:y:0*
T0*1
_output_shapes
:������������
6tf.keras.backend.binary_crossentropy/logistic_loss/NegNeg+model_1/conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:������������
;tf.keras.backend.binary_crossentropy/logistic_loss/Select_1SelectCtf.keras.backend.binary_crossentropy/logistic_loss/GreaterEqual:z:0:tf.keras.backend.binary_crossentropy/logistic_loss/Neg:y:0+model_1/conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:������������
6tf.keras.backend.binary_crossentropy/logistic_loss/mulMul+model_1/conv2d_transpose_3/BiasAdd:output:0inputs*
T0*1
_output_shapes
:������������
6tf.keras.backend.binary_crossentropy/logistic_loss/subSubBtf.keras.backend.binary_crossentropy/logistic_loss/Select:output:0:tf.keras.backend.binary_crossentropy/logistic_loss/mul:z:0*
T0*1
_output_shapes
:������������
6tf.keras.backend.binary_crossentropy/logistic_loss/ExpExpDtf.keras.backend.binary_crossentropy/logistic_loss/Select_1:output:0*
T0*1
_output_shapes
:������������
8tf.keras.backend.binary_crossentropy/logistic_loss/Log1pLog1p:tf.keras.backend.binary_crossentropy/logistic_loss/Exp:y:0*
T0*1
_output_shapes
:������������
2tf.keras.backend.binary_crossentropy/logistic_lossAddV2:tf.keras.backend.binary_crossentropy/logistic_loss/sub:z:0<tf.keras.backend.binary_crossentropy/logistic_loss/Log1p:y:0*
T0*1
_output_shapes
:�����������j
tf.math.exp_1/ExpExpmodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
tf.__operators__.add_1/AddV2AddV2unknownmodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@o
tf.math.square/SquareSquaremodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@}
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         �
tf.math.reduce_sum/SumSum6tf.keras.backend.binary_crossentropy/logistic_loss:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:����������
tf.math.subtract/SubSub tf.__operators__.add_1/AddV2:z:0tf.math.square/Square:y:0*
T0*'
_output_shapes
:���������@�
tf.math.subtract_1/SubSubtf.math.subtract/Sub:z:0tf.math.exp_1/Exp:y:0*
T0*'
_output_shapes
:���������@e
tf.math.reduce_mean_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_mean_1/MeanMeantf.math.reduce_sum/Sum:output:0$tf.math.reduce_mean_1/Const:output:0*
T0*
_output_shapes
: u
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
tf.math.reduce_sum_1/SumSumtf.math.subtract_1/Sub:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:���������Q
add_metric/RankConst*
_output_shapes
: *
dtype0*
value	B : X
add_metric/range/startConst*
_output_shapes
: *
dtype0*
value	B : X
add_metric/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
add_metric/rangeRangeadd_metric/range/start:output:0add_metric/Rank:output:0add_metric/range/delta:output:0*
_output_shapes
: v
add_metric/SumSum#tf.math.reduce_mean_1/Mean:output:0add_metric/range:output:0*
T0*
_output_shapes
: �
add_metric/AssignAddVariableOpAssignAddVariableOp'add_metric_assignaddvariableop_resourceadd_metric/Sum:output:0*
_output_shapes
 *
dtype0Q
add_metric/SizeConst*
_output_shapes
: *
dtype0*
value	B :a
add_metric/CastCastadd_metric/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: �
 add_metric/AssignAddVariableOp_1AssignAddVariableOp)add_metric_assignaddvariableop_1_resourceadd_metric/Cast:y:0^add_metric/AssignAddVariableOp*
_output_shapes
 *
dtype0�
$add_metric/div_no_nan/ReadVariableOpReadVariableOp'add_metric_assignaddvariableop_resource^add_metric/AssignAddVariableOp!^add_metric/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
&add_metric/div_no_nan/ReadVariableOp_1ReadVariableOp)add_metric_assignaddvariableop_1_resource!^add_metric/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
add_metric/div_no_nanDivNoNan,add_metric/div_no_nan/ReadVariableOp:value:0.add_metric/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: [
add_metric/IdentityIdentityadd_metric/div_no_nan:z:0*
T0*
_output_shapes
: y
tf.math.multiply_1/MulMul	unknown_0!tf.math.reduce_sum_1/Sum:output:0*
T0*#
_output_shapes
:����������
tf.__operators__.add_2/AddV2AddV2tf.math.reduce_sum/Sum:output:0tf.math.multiply_1/Mul:z:0*
T0*#
_output_shapes
:���������e
tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_mean_2/MeanMeantf.math.multiply_1/Mul:z:0$tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: c
tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_mean/MeanMean tf.__operators__.add_2/AddV2:z:0"tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: S
add_metric_1/RankConst*
_output_shapes
: *
dtype0*
value	B : Z
add_metric_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : Z
add_metric_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
add_metric_1/rangeRange!add_metric_1/range/start:output:0add_metric_1/Rank:output:0!add_metric_1/range/delta:output:0*
_output_shapes
: z
add_metric_1/SumSum#tf.math.reduce_mean_2/Mean:output:0add_metric_1/range:output:0*
T0*
_output_shapes
: �
 add_metric_1/AssignAddVariableOpAssignAddVariableOp)add_metric_1_assignaddvariableop_resourceadd_metric_1/Sum:output:0*
_output_shapes
 *
dtype0S
add_metric_1/SizeConst*
_output_shapes
: *
dtype0*
value	B :e
add_metric_1/CastCastadd_metric_1/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"add_metric_1/AssignAddVariableOp_1AssignAddVariableOp+add_metric_1_assignaddvariableop_1_resourceadd_metric_1/Cast:y:0!^add_metric_1/AssignAddVariableOp*
_output_shapes
 *
dtype0�
&add_metric_1/div_no_nan/ReadVariableOpReadVariableOp)add_metric_1_assignaddvariableop_resource!^add_metric_1/AssignAddVariableOp#^add_metric_1/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
(add_metric_1/div_no_nan/ReadVariableOp_1ReadVariableOp+add_metric_1_assignaddvariableop_1_resource#^add_metric_1/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
add_metric_1/div_no_nanDivNoNan.add_metric_1/div_no_nan/ReadVariableOp:value:00add_metric_1/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: _
add_metric_1/IdentityIdentityadd_metric_1/div_no_nan:z:0*
T0*
_output_shapes
: S
add_metric_2/RankConst*
_output_shapes
: *
dtype0*
value	B : Z
add_metric_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : Z
add_metric_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
add_metric_2/rangeRange!add_metric_2/range/start:output:0add_metric_2/Rank:output:0!add_metric_2/range/delta:output:0*
_output_shapes
: x
add_metric_2/SumSum!tf.math.reduce_mean/Mean:output:0add_metric_2/range:output:0*
T0*
_output_shapes
: �
 add_metric_2/AssignAddVariableOpAssignAddVariableOp)add_metric_2_assignaddvariableop_resourceadd_metric_2/Sum:output:0*
_output_shapes
 *
dtype0S
add_metric_2/SizeConst*
_output_shapes
: *
dtype0*
value	B :e
add_metric_2/CastCastadd_metric_2/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"add_metric_2/AssignAddVariableOp_1AssignAddVariableOp+add_metric_2_assignaddvariableop_1_resourceadd_metric_2/Cast:y:0!^add_metric_2/AssignAddVariableOp*
_output_shapes
 *
dtype0�
&add_metric_2/div_no_nan/ReadVariableOpReadVariableOp)add_metric_2_assignaddvariableop_resource!^add_metric_2/AssignAddVariableOp#^add_metric_2/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
(add_metric_2/div_no_nan/ReadVariableOp_1ReadVariableOp+add_metric_2_assignaddvariableop_1_resource#^add_metric_2/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
add_metric_2/div_no_nanDivNoNan.add_metric_2/div_no_nan/ReadVariableOp:value:00add_metric_2/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: _
add_metric_2/IdentityIdentityadd_metric_2/div_no_nan:z:0*
T0*
_output_shapes
: 
IdentityIdentity&model_1/conv2d_transpose_3/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:�����������a

Identity_1Identity!tf.math.reduce_mean/Mean:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^add_metric/AssignAddVariableOp!^add_metric/AssignAddVariableOp_1%^add_metric/div_no_nan/ReadVariableOp'^add_metric/div_no_nan/ReadVariableOp_1!^add_metric_1/AssignAddVariableOp#^add_metric_1/AssignAddVariableOp_1'^add_metric_1/div_no_nan/ReadVariableOp)^add_metric_1/div_no_nan/ReadVariableOp_1!^add_metric_2/AssignAddVariableOp#^add_metric_2/AssignAddVariableOp_1'^add_metric_2/div_no_nan/ReadVariableOp)^add_metric_2/div_no_nan/ReadVariableOp_1$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp0^model_1/conv2d_transpose/BiasAdd/ReadVariableOp9^model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2^model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2^model_1/conv2d_transpose_2/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2^model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp'^model_1/dense_2/BiasAdd/ReadVariableOp&^model_1/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
add_metric/AssignAddVariableOpadd_metric/AssignAddVariableOp2D
 add_metric/AssignAddVariableOp_1 add_metric/AssignAddVariableOp_12L
$add_metric/div_no_nan/ReadVariableOp$add_metric/div_no_nan/ReadVariableOp2P
&add_metric/div_no_nan/ReadVariableOp_1&add_metric/div_no_nan/ReadVariableOp_12D
 add_metric_1/AssignAddVariableOp add_metric_1/AssignAddVariableOp2H
"add_metric_1/AssignAddVariableOp_1"add_metric_1/AssignAddVariableOp_12P
&add_metric_1/div_no_nan/ReadVariableOp&add_metric_1/div_no_nan/ReadVariableOp2T
(add_metric_1/div_no_nan/ReadVariableOp_1(add_metric_1/div_no_nan/ReadVariableOp_12D
 add_metric_2/AssignAddVariableOp add_metric_2/AssignAddVariableOp2H
"add_metric_2/AssignAddVariableOp_1"add_metric_2/AssignAddVariableOp_12P
&add_metric_2/div_no_nan/ReadVariableOp&add_metric_2/div_no_nan/ReadVariableOp2T
(add_metric_2/div_no_nan/ReadVariableOp_1(add_metric_2/div_no_nan/ReadVariableOp_12J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2b
/model_1/conv2d_transpose/BiasAdd/ReadVariableOp/model_1/conv2d_transpose/BiasAdd/ReadVariableOp2t
8model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp8model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2f
1model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2f
1model_1/conv2d_transpose_2/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2f
1model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2P
&model_1/dense_2/BiasAdd/ReadVariableOp&model_1/dense_2/BiasAdd/ReadVariableOp2N
%model_1/dense_2/MatMul/ReadVariableOp%model_1/dense_2/MatMul/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�!
�
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_241781

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
_
C__inference_reshape_layer_call_and_return_conditional_losses_239314

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : �
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:��������� `
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:�����������:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�"
�
C__inference_model_1_layer_call_and_return_conditional_losses_239541
input_2"
dense_2_239514:
@��
dense_2_239516:
��1
conv2d_transpose_239520:  %
conv2d_transpose_239522: 3
conv2d_transpose_1_239525: '
conv2d_transpose_1_239527:3
conv2d_transpose_2_239530:'
conv2d_transpose_2_239532:3
conv2d_transpose_3_239535:'
conv2d_transpose_3_239537:
identity��(conv2d_transpose/StatefulPartitionedCall�*conv2d_transpose_1/StatefulPartitionedCall�*conv2d_transpose_2/StatefulPartitionedCall�*conv2d_transpose_3/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_2_239514dense_2_239516*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_239294�
reshape/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_239314�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_239520conv2d_transpose_239522*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_239134�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_239525conv2d_transpose_1_239527*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_239179�
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_239530conv2d_transpose_2_239532*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_239224�
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_239535conv2d_transpose_3_239537*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_239269�
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
'
_output_shapes
:���������@
!
_user_specified_name	input_2
�
�
A__inference_model_layer_call_and_return_conditional_losses_239065
input_1'
conv2d_239037:
conv2d_239039:)
conv2d_1_239042:
conv2d_1_239044:)
conv2d_2_239047: 
conv2d_2_239049: "
dense_1_239053:
��@
dense_1_239055:@ 
dense_239058:
��@
dense_239060:@
identity

identity_1��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_239037conv2d_239039*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_238759�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_239042conv2d_1_239044*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_238776�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_239047conv2d_2_239049*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_238793�
flatten/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_238805�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_239053dense_1_239055*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_238817�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_239058dense_239060*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_238833u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@y

Identity_1Identity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�/
�
A__inference_model_layer_call_and_return_conditional_losses_241123

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource: :
&dense_1_matmul_readvariableop_resource:
��@5
'dense_1_biasadd_readvariableop_resource:@8
$dense_matmul_readvariableop_resource:
��@3
%dense_biasadd_readvariableop_resource:@
identity

identity_1��conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ppf
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������88�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:��������� ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� b  �
flatten/ReshapeReshapeconv2d_2/Relu:activations:0flatten/Const:output:0*
T0*)
_output_shapes
:������������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��@*
dtype0�
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��@*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@i

Identity_1Identitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
&__inference_dense_layer_call_fn_241584

inputs
unknown:
��@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_238833o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
_
C__inference_reshape_layer_call_and_return_conditional_losses_241652

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : �
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:��������� `
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:�����������:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_241056

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
��@
	unknown_6:@
	unknown_7:
��@
	unknown_8:@
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_238841o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
(__inference_model_1_layer_call_fn_239481
input_2
unknown:
@��
	unknown_0:
��#
	unknown_1:  
	unknown_2: #
	unknown_3: 
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_239433y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������@
!
_user_specified_name	input_2
�i
�	
C__inference_model_2_layer_call_and_return_conditional_losses_240260
input_1&
model_240139:
model_240141:&
model_240143:
model_240145:&
model_240147: 
model_240149:  
model_240151:
��@
model_240153:@ 
model_240155:
��@
model_240157:@"
model_1_240180:
@��
model_1_240182:
��(
model_1_240184:  
model_1_240186: (
model_1_240188: 
model_1_240190:(
model_1_240192:
model_1_240194:(
model_1_240196:
model_1_240198:
unknown
add_metric_240233: 
add_metric_240235: 
	unknown_0
add_metric_1_240246: 
add_metric_1_240248: 
add_metric_2_240253: 
add_metric_2_240255: 
identity

identity_1��"add_metric/StatefulPartitionedCall�$add_metric_1/StatefulPartitionedCall�$add_metric_2/StatefulPartitionedCall�model/StatefulPartitionedCall�model_1/StatefulPartitionedCall�
model/StatefulPartitionedCallStatefulPartitionedCallinput_1model_240139model_240141model_240143model_240145model_240147model_240149model_240151model_240153model_240155model_240157*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_238841p
tf.compat.v1.shape_1/ShapeShape&model/StatefulPartitionedCall:output:1*
T0*
_output_shapes
:n
tf.compat.v1.shape/ShapeShape&model/StatefulPartitionedCall:output:1*
T0*
_output_shapes
:v
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&tf.__operators__.getitem/strided_sliceStridedSlice!tf.compat.v1.shape/Shape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(tf.__operators__.getitem_1/strided_sliceStridedSlice#tf.compat.v1.shape_1/Shape:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
tf.math.exp/ExpExp&model/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:���������@�
$tf.random.normal/random_normal/shapePack/tf.__operators__.getitem/strided_slice:output:01tf.__operators__.getitem_1/strided_slice:output:0*
N*
T0*
_output_shapes
:h
#tf.random.normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%tf.random.normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
3tf.random.normal/random_normal/RandomStandardNormalRandomStandardNormal-tf.random.normal/random_normal/shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0�
"tf.random.normal/random_normal/mulMul<tf.random.normal/random_normal/RandomStandardNormal:output:0.tf.random.normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:���������@�
tf.random.normal/random_normalAddV2&tf.random.normal/random_normal/mul:z:0,tf.random.normal/random_normal/mean:output:0*
T0*'
_output_shapes
:���������@�
tf.math.multiply/MulMultf.math.exp/Exp:y:0"tf.random.normal/random_normal:z:0*
T0*'
_output_shapes
:���������@�
tf.__operators__.add/AddV2AddV2&model/StatefulPartitionedCall:output:0tf.math.multiply/Mul:z:0*
T0*'
_output_shapes
:���������@�
model_1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0model_1_240180model_1_240182model_1_240184model_1_240186model_1_240188model_1_240190model_1_240192model_1_240194model_1_240196model_1_240198*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_239337o
*tf.keras.backend.binary_crossentropy/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���3o
*tf.keras.backend.binary_crossentropy/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
(tf.keras.backend.binary_crossentropy/subSub3tf.keras.backend.binary_crossentropy/sub/x:output:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*
_output_shapes
: �
:tf.keras.backend.binary_crossentropy/clip_by_value/MinimumMinimum(model_1/StatefulPartitionedCall:output:0,tf.keras.backend.binary_crossentropy/sub:z:0*
T0*1
_output_shapes
:������������
2tf.keras.backend.binary_crossentropy/clip_by_valueMaximum>tf.keras.backend.binary_crossentropy/clip_by_value/Minimum:z:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*1
_output_shapes
:�����������o
*tf.keras.backend.binary_crossentropy/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
(tf.keras.backend.binary_crossentropy/addAddV26tf.keras.backend.binary_crossentropy/clip_by_value:z:03tf.keras.backend.binary_crossentropy/add/y:output:0*
T0*1
_output_shapes
:������������
(tf.keras.backend.binary_crossentropy/LogLog,tf.keras.backend.binary_crossentropy/add:z:0*
T0*1
_output_shapes
:������������
(tf.keras.backend.binary_crossentropy/mulMulinput_1,tf.keras.backend.binary_crossentropy/Log:y:0*
T0*1
_output_shapes
:�����������q
,tf.keras.backend.binary_crossentropy/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
*tf.keras.backend.binary_crossentropy/sub_1Sub5tf.keras.backend.binary_crossentropy/sub_1/x:output:0input_1*
T0*1
_output_shapes
:�����������q
,tf.keras.backend.binary_crossentropy/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
*tf.keras.backend.binary_crossentropy/sub_2Sub5tf.keras.backend.binary_crossentropy/sub_2/x:output:06tf.keras.backend.binary_crossentropy/clip_by_value:z:0*
T0*1
_output_shapes
:�����������q
,tf.keras.backend.binary_crossentropy/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
*tf.keras.backend.binary_crossentropy/add_1AddV2.tf.keras.backend.binary_crossentropy/sub_2:z:05tf.keras.backend.binary_crossentropy/add_1/y:output:0*
T0*1
_output_shapes
:������������
*tf.keras.backend.binary_crossentropy/Log_1Log.tf.keras.backend.binary_crossentropy/add_1:z:0*
T0*1
_output_shapes
:������������
*tf.keras.backend.binary_crossentropy/mul_1Mul.tf.keras.backend.binary_crossentropy/sub_1:z:0.tf.keras.backend.binary_crossentropy/Log_1:y:0*
T0*1
_output_shapes
:������������
*tf.keras.backend.binary_crossentropy/add_2AddV2,tf.keras.backend.binary_crossentropy/mul:z:0.tf.keras.backend.binary_crossentropy/mul_1:z:0*
T0*1
_output_shapes
:������������
(tf.keras.backend.binary_crossentropy/NegNeg.tf.keras.backend.binary_crossentropy/add_2:z:0*
T0*1
_output_shapes
:�����������r
tf.math.exp_1/ExpExp&model/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:���������@�
tf.__operators__.add_1/AddV2AddV2unknown&model/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:���������@y
tf.math.square/SquareSquare&model/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@}
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         �
tf.math.reduce_sum/SumSum,tf.keras.backend.binary_crossentropy/Neg:y:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:����������
tf.math.subtract/SubSub tf.__operators__.add_1/AddV2:z:0tf.math.square/Square:y:0*
T0*'
_output_shapes
:���������@�
tf.math.subtract_1/SubSubtf.math.subtract/Sub:z:0tf.math.exp_1/Exp:y:0*
T0*'
_output_shapes
:���������@e
tf.math.reduce_mean_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_mean_1/MeanMeantf.math.reduce_sum/Sum:output:0$tf.math.reduce_mean_1/Const:output:0*
T0*
_output_shapes
: u
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
tf.math.reduce_sum_1/SumSumtf.math.subtract_1/Sub:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:����������
"add_metric/StatefulPartitionedCallStatefulPartitionedCall#tf.math.reduce_mean_1/Mean:output:0add_metric_240233add_metric_240235*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_add_metric_layer_call_and_return_conditional_losses_239659y
tf.math.multiply_1/MulMul	unknown_0!tf.math.reduce_sum_1/Sum:output:0*
T0*#
_output_shapes
:����������
tf.__operators__.add_2/AddV2AddV2tf.math.reduce_sum/Sum:output:0tf.math.multiply_1/Mul:z:0*
T0*#
_output_shapes
:���������e
tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_mean_2/MeanMeantf.math.multiply_1/Mul:z:0$tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: c
tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_mean/MeanMean tf.__operators__.add_2/AddV2:z:0"tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: �
$add_metric_1/StatefulPartitionedCallStatefulPartitionedCall#tf.math.reduce_mean_2/Mean:output:0add_metric_1_240246add_metric_1_240248*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_add_metric_1_layer_call_and_return_conditional_losses_239690�
add_loss/PartitionedCallPartitionedCall!tf.math.reduce_mean/Mean:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_add_loss_layer_call_and_return_conditional_losses_239701�
$add_metric_2/StatefulPartitionedCallStatefulPartitionedCall!tf.math.reduce_mean/Mean:output:0add_metric_2_240253add_metric_2_240255*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_add_metric_2_layer_call_and_return_conditional_losses_239721�
IdentityIdentity(model_1/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������a

Identity_1Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: �
NoOpNoOp#^add_metric/StatefulPartitionedCall%^add_metric_1/StatefulPartitionedCall%^add_metric_2/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"add_metric/StatefulPartitionedCall"add_metric/StatefulPartitionedCall2L
$add_metric_1/StatefulPartitionedCall$add_metric_1/StatefulPartitionedCall2L
$add_metric_2/StatefulPartitionedCall$add_metric_2/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: 
��
�
__inference__traced_save_242068
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop8
4savev2_conv2d_transpose_2_kernel_read_readvariableop6
2savev2_conv2d_transpose_2_bias_read_readvariableop8
4savev2_conv2d_transpose_3_kernel_read_readvariableop6
2savev2_conv2d_transpose_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop/
+savev2_add_metric_total_read_readvariableop/
+savev2_add_metric_count_read_readvariableop1
-savev2_add_metric_1_total_read_readvariableop1
-savev2_add_metric_1_count_read_readvariableop1
-savev2_add_metric_2_total_read_readvariableop1
-savev2_add_metric_2_count_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop=
9savev2_adam_conv2d_transpose_kernel_m_read_readvariableop;
7savev2_adam_conv2d_transpose_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_1_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_1_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_2_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_2_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_3_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_3_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop=
9savev2_adam_conv2d_transpose_kernel_v_read_readvariableop;
7savev2_adam_conv2d_transpose_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_1_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_1_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_2_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_2_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_3_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_3_bias_v_read_readvariableop
savev2_const_2

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �!
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*�!
value�!B�!JB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*�
value�B�JB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop4savev2_conv2d_transpose_2_kernel_read_readvariableop2savev2_conv2d_transpose_2_bias_read_readvariableop4savev2_conv2d_transpose_3_kernel_read_readvariableop2savev2_conv2d_transpose_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop+savev2_add_metric_total_read_readvariableop+savev2_add_metric_count_read_readvariableop-savev2_add_metric_1_total_read_readvariableop-savev2_add_metric_1_count_read_readvariableop-savev2_add_metric_2_total_read_readvariableop-savev2_add_metric_2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop9savev2_adam_conv2d_transpose_kernel_m_read_readvariableop7savev2_adam_conv2d_transpose_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_1_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_1_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_2_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_2_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_3_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_3_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop9savev2_adam_conv2d_transpose_kernel_v_read_readvariableop7savev2_adam_conv2d_transpose_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_1_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_1_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_2_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_2_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_3_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_3_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *X
dtypesN
L2J	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: ::::: : :
��@:@:
��@:@:
@��:��:  : : :::::: : : : : : : : : : : : : ::::: : :
��@:@:
��@:@:
@��:��:  : : :::::::::: : :
��@:@:
��@:@:
@��:��:  : : :::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
��@: 

_output_shapes
:@:&	"
 
_output_shapes
:
��@: 


_output_shapes
:@:&"
 
_output_shapes
:
@��:"

_output_shapes

:��:,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :,"(
&
_output_shapes
:: #

_output_shapes
::,$(
&
_output_shapes
:: %

_output_shapes
::,&(
&
_output_shapes
: : '

_output_shapes
: :&("
 
_output_shapes
:
��@: )

_output_shapes
:@:&*"
 
_output_shapes
:
��@: +

_output_shapes
:@:&,"
 
_output_shapes
:
@��:"-

_output_shapes

:��:,.(
&
_output_shapes
:  : /

_output_shapes
: :,0(
&
_output_shapes
: : 1

_output_shapes
::,2(
&
_output_shapes
:: 3

_output_shapes
::,4(
&
_output_shapes
:: 5

_output_shapes
::,6(
&
_output_shapes
:: 7

_output_shapes
::,8(
&
_output_shapes
:: 9

_output_shapes
::,:(
&
_output_shapes
: : ;

_output_shapes
: :&<"
 
_output_shapes
:
��@: =

_output_shapes
:@:&>"
 
_output_shapes
:
��@: ?

_output_shapes
:@:&@"
 
_output_shapes
:
@��:"A

_output_shapes

:��:,B(
&
_output_shapes
:  : C

_output_shapes
: :,D(
&
_output_shapes
: : E

_output_shapes
::,F(
&
_output_shapes
:: G

_output_shapes
::,H(
&
_output_shapes
:: I

_output_shapes
::J

_output_shapes
: 
��
�	
C__inference_model_1_layer_call_and_return_conditional_losses_241314

inputs:
&dense_2_matmul_readvariableop_resource:
@��7
'dense_2_biasadd_readvariableop_resource:
��S
9conv2d_transpose_conv2d_transpose_readvariableop_resource:  >
0conv2d_transpose_biasadd_readvariableop_resource: U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_1_biasadd_readvariableop_resource:U
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_2_biasadd_readvariableop_resource:U
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_3_biasadd_readvariableop_resource:
identity��'conv2d_transpose/BiasAdd/ReadVariableOp�0conv2d_transpose/conv2d_transpose/ReadVariableOp�)conv2d_transpose_1/BiasAdd/ReadVariableOp�2conv2d_transpose_1/conv2d_transpose/ReadVariableOp�)conv2d_transpose_2/BiasAdd/ReadVariableOp�2conv2d_transpose_2/conv2d_transpose/ReadVariableOp�)conv2d_transpose_3/BiasAdd/ReadVariableOp�2conv2d_transpose_3/conv2d_transpose/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
@��*
dtype0{
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:������������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes

:��*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������b
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*)
_output_shapes
:�����������W
reshape/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : �
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape/ReshapeReshapedense_2/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� ^
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :8Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :8Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0�
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:���������88 *
paddingSAME*
strides
�
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88 z
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:���������88 k
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :p\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :p\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp~
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������ppm
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�]
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�\
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0�
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
conv2d_transpose_2/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*1
_output_shapes
:�����������m
conv2d_transpose_3/ShapeShape%conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�]
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0�
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_2/Relu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
conv2d_transpose_3/SigmoidSigmoid#conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:�����������w
IdentityIdentityconv2d_transpose_3/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
3__inference_conv2d_transpose_2_layer_call_fn_241747

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_239224�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
'__inference_conv2d_layer_call_fn_241513

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_238759w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������pp`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
H__inference_add_metric_1_layer_call_and_return_conditional_losses_241478

inputs&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: 

identity_1��AssignAddVariableOp�AssignAddVariableOp_1�div_no_nan/ReadVariableOp�div_no_nan/ReadVariableOp_1F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: C
SumSuminputsrange:output:0*
T0*
_output_shapes
: y
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: �
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0�
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: F

Identity_1Identityinputs^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_126
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_1:> :

_output_shapes
: 
 
_user_specified_nameinputs
��
�
!__inference__wrapped_model_238741
input_1M
3model_2_model_conv2d_conv2d_readvariableop_resource:B
4model_2_model_conv2d_biasadd_readvariableop_resource:O
5model_2_model_conv2d_1_conv2d_readvariableop_resource:D
6model_2_model_conv2d_1_biasadd_readvariableop_resource:O
5model_2_model_conv2d_2_conv2d_readvariableop_resource: D
6model_2_model_conv2d_2_biasadd_readvariableop_resource: H
4model_2_model_dense_1_matmul_readvariableop_resource:
��@C
5model_2_model_dense_1_biasadd_readvariableop_resource:@F
2model_2_model_dense_matmul_readvariableop_resource:
��@A
3model_2_model_dense_biasadd_readvariableop_resource:@J
6model_2_model_1_dense_2_matmul_readvariableop_resource:
@��G
7model_2_model_1_dense_2_biasadd_readvariableop_resource:
��c
Imodel_2_model_1_conv2d_transpose_conv2d_transpose_readvariableop_resource:  N
@model_2_model_1_conv2d_transpose_biasadd_readvariableop_resource: e
Kmodel_2_model_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource: P
Bmodel_2_model_1_conv2d_transpose_1_biasadd_readvariableop_resource:e
Kmodel_2_model_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:P
Bmodel_2_model_1_conv2d_transpose_2_biasadd_readvariableop_resource:e
Kmodel_2_model_1_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:P
Bmodel_2_model_1_conv2d_transpose_3_biasadd_readvariableop_resource:
model_2_2386819
/model_2_add_metric_assignaddvariableop_resource: ;
1model_2_add_metric_assignaddvariableop_1_resource: 
model_2_238706;
1model_2_add_metric_1_assignaddvariableop_resource: =
3model_2_add_metric_1_assignaddvariableop_1_resource: ;
1model_2_add_metric_2_assignaddvariableop_resource: =
3model_2_add_metric_2_assignaddvariableop_1_resource: 
identity��&model_2/add_metric/AssignAddVariableOp�(model_2/add_metric/AssignAddVariableOp_1�,model_2/add_metric/div_no_nan/ReadVariableOp�.model_2/add_metric/div_no_nan/ReadVariableOp_1�(model_2/add_metric_1/AssignAddVariableOp�*model_2/add_metric_1/AssignAddVariableOp_1�.model_2/add_metric_1/div_no_nan/ReadVariableOp�0model_2/add_metric_1/div_no_nan/ReadVariableOp_1�(model_2/add_metric_2/AssignAddVariableOp�*model_2/add_metric_2/AssignAddVariableOp_1�.model_2/add_metric_2/div_no_nan/ReadVariableOp�0model_2/add_metric_2/div_no_nan/ReadVariableOp_1�+model_2/model/conv2d/BiasAdd/ReadVariableOp�*model_2/model/conv2d/Conv2D/ReadVariableOp�-model_2/model/conv2d_1/BiasAdd/ReadVariableOp�,model_2/model/conv2d_1/Conv2D/ReadVariableOp�-model_2/model/conv2d_2/BiasAdd/ReadVariableOp�,model_2/model/conv2d_2/Conv2D/ReadVariableOp�*model_2/model/dense/BiasAdd/ReadVariableOp�)model_2/model/dense/MatMul/ReadVariableOp�,model_2/model/dense_1/BiasAdd/ReadVariableOp�+model_2/model/dense_1/MatMul/ReadVariableOp�7model_2/model_1/conv2d_transpose/BiasAdd/ReadVariableOp�@model_2/model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp�9model_2/model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp�Bmodel_2/model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp�9model_2/model_1/conv2d_transpose_2/BiasAdd/ReadVariableOp�Bmodel_2/model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp�9model_2/model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp�Bmodel_2/model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp�.model_2/model_1/dense_2/BiasAdd/ReadVariableOp�-model_2/model_1/dense_2/MatMul/ReadVariableOp�
*model_2/model/conv2d/Conv2D/ReadVariableOpReadVariableOp3model_2_model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
model_2/model/conv2d/Conv2DConv2Dinput_12model_2/model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
+model_2/model/conv2d/BiasAdd/ReadVariableOpReadVariableOp4model_2_model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_2/model/conv2d/BiasAddBiasAdd$model_2/model/conv2d/Conv2D:output:03model_2/model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp�
model_2/model/conv2d/ReluRelu%model_2/model/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp�
,model_2/model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5model_2_model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
model_2/model/conv2d_1/Conv2DConv2D'model_2/model/conv2d/Relu:activations:04model_2/model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
-model_2/model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6model_2_model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_2/model/conv2d_1/BiasAddBiasAdd&model_2/model/conv2d_1/Conv2D:output:05model_2/model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88�
model_2/model/conv2d_1/ReluRelu'model_2/model/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������88�
,model_2/model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp5model_2_model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
model_2/model/conv2d_2/Conv2DConv2D)model_2/model/conv2d_1/Relu:activations:04model_2/model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
-model_2/model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp6model_2_model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_2/model/conv2d_2/BiasAddBiasAdd&model_2/model/conv2d_2/Conv2D:output:05model_2/model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
model_2/model/conv2d_2/ReluRelu'model_2/model/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:��������� l
model_2/model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� b  �
model_2/model/flatten/ReshapeReshape)model_2/model/conv2d_2/Relu:activations:0$model_2/model/flatten/Const:output:0*
T0*)
_output_shapes
:������������
+model_2/model/dense_1/MatMul/ReadVariableOpReadVariableOp4model_2_model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��@*
dtype0�
model_2/model/dense_1/MatMulMatMul&model_2/model/flatten/Reshape:output:03model_2/model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,model_2/model/dense_1/BiasAdd/ReadVariableOpReadVariableOp5model_2_model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_2/model/dense_1/BiasAddBiasAdd&model_2/model/dense_1/MatMul:product:04model_2/model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)model_2/model/dense/MatMul/ReadVariableOpReadVariableOp2model_2_model_dense_matmul_readvariableop_resource* 
_output_shapes
:
��@*
dtype0�
model_2/model/dense/MatMulMatMul&model_2/model/flatten/Reshape:output:01model_2/model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*model_2/model/dense/BiasAdd/ReadVariableOpReadVariableOp3model_2_model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_2/model/dense/BiasAddBiasAdd$model_2/model/dense/MatMul:product:02model_2/model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@x
"model_2/tf.compat.v1.shape_1/ShapeShape&model_2/model/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:v
 model_2/tf.compat.v1.shape/ShapeShape&model_2/model/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:~
4model_2/tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6model_2/tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6model_2/tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.model_2/tf.__operators__.getitem/strided_sliceStridedSlice)model_2/tf.compat.v1.shape/Shape:output:0=model_2/tf.__operators__.getitem/strided_slice/stack:output:0?model_2/tf.__operators__.getitem/strided_slice/stack_1:output:0?model_2/tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
6model_2/tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8model_2/tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
8model_2/tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0model_2/tf.__operators__.getitem_1/strided_sliceStridedSlice+model_2/tf.compat.v1.shape_1/Shape:output:0?model_2/tf.__operators__.getitem_1/strided_slice/stack:output:0Amodel_2/tf.__operators__.getitem_1/strided_slice/stack_1:output:0Amodel_2/tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
model_2/tf.math.exp/ExpExp&model_2/model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
,model_2/tf.random.normal/random_normal/shapePack7model_2/tf.__operators__.getitem/strided_slice:output:09model_2/tf.__operators__.getitem_1/strided_slice:output:0*
N*
T0*
_output_shapes
:p
+model_2/tf.random.normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    r
-model_2/tf.random.normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
;model_2/tf.random.normal/random_normal/RandomStandardNormalRandomStandardNormal5model_2/tf.random.normal/random_normal/shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0�
*model_2/tf.random.normal/random_normal/mulMulDmodel_2/tf.random.normal/random_normal/RandomStandardNormal:output:06model_2/tf.random.normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:���������@�
&model_2/tf.random.normal/random_normalAddV2.model_2/tf.random.normal/random_normal/mul:z:04model_2/tf.random.normal/random_normal/mean:output:0*
T0*'
_output_shapes
:���������@�
model_2/tf.math.multiply/MulMulmodel_2/tf.math.exp/Exp:y:0*model_2/tf.random.normal/random_normal:z:0*
T0*'
_output_shapes
:���������@�
"model_2/tf.__operators__.add/AddV2AddV2$model_2/model/dense/BiasAdd:output:0 model_2/tf.math.multiply/Mul:z:0*
T0*'
_output_shapes
:���������@�
-model_2/model_1/dense_2/MatMul/ReadVariableOpReadVariableOp6model_2_model_1_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
@��*
dtype0�
model_2/model_1/dense_2/MatMulMatMul&model_2/tf.__operators__.add/AddV2:z:05model_2/model_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:������������
.model_2/model_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp7model_2_model_1_dense_2_biasadd_readvariableop_resource*
_output_shapes

:��*
dtype0�
model_2/model_1/dense_2/BiasAddBiasAdd(model_2/model_1/dense_2/MatMul:product:06model_2/model_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:������������
model_2/model_1/dense_2/ReluRelu(model_2/model_1/dense_2/BiasAdd:output:0*
T0*)
_output_shapes
:�����������w
model_2/model_1/reshape/ShapeShape*model_2/model_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:u
+model_2/model_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-model_2/model_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-model_2/model_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%model_2/model_1/reshape/strided_sliceStridedSlice&model_2/model_1/reshape/Shape:output:04model_2/model_1/reshape/strided_slice/stack:output:06model_2/model_1/reshape/strided_slice/stack_1:output:06model_2/model_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'model_2/model_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :i
'model_2/model_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :i
'model_2/model_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : �
%model_2/model_1/reshape/Reshape/shapePack.model_2/model_1/reshape/strided_slice:output:00model_2/model_1/reshape/Reshape/shape/1:output:00model_2/model_1/reshape/Reshape/shape/2:output:00model_2/model_1/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
model_2/model_1/reshape/ReshapeReshape*model_2/model_1/dense_2/Relu:activations:0.model_2/model_1/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� ~
&model_2/model_1/conv2d_transpose/ShapeShape(model_2/model_1/reshape/Reshape:output:0*
T0*
_output_shapes
:~
4model_2/model_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6model_2/model_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6model_2/model_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.model_2/model_1/conv2d_transpose/strided_sliceStridedSlice/model_2/model_1/conv2d_transpose/Shape:output:0=model_2/model_1/conv2d_transpose/strided_slice/stack:output:0?model_2/model_1/conv2d_transpose/strided_slice/stack_1:output:0?model_2/model_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model_2/model_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :8j
(model_2/model_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :8j
(model_2/model_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
&model_2/model_1/conv2d_transpose/stackPack7model_2/model_1/conv2d_transpose/strided_slice:output:01model_2/model_1/conv2d_transpose/stack/1:output:01model_2/model_1/conv2d_transpose/stack/2:output:01model_2/model_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:�
6model_2/model_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
8model_2/model_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
8model_2/model_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0model_2/model_1/conv2d_transpose/strided_slice_1StridedSlice/model_2/model_1/conv2d_transpose/stack:output:0?model_2/model_1/conv2d_transpose/strided_slice_1/stack:output:0Amodel_2/model_1/conv2d_transpose/strided_slice_1/stack_1:output:0Amodel_2/model_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
@model_2/model_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpImodel_2_model_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0�
1model_2/model_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput/model_2/model_1/conv2d_transpose/stack:output:0Hmodel_2/model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0(model_2/model_1/reshape/Reshape:output:0*
T0*/
_output_shapes
:���������88 *
paddingSAME*
strides
�
7model_2/model_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp@model_2_model_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
(model_2/model_1/conv2d_transpose/BiasAddBiasAdd:model_2/model_1/conv2d_transpose/conv2d_transpose:output:0?model_2/model_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88 �
%model_2/model_1/conv2d_transpose/ReluRelu1model_2/model_1/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:���������88 �
(model_2/model_1/conv2d_transpose_1/ShapeShape3model_2/model_1/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:�
6model_2/model_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
8model_2/model_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
8model_2/model_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0model_2/model_1/conv2d_transpose_1/strided_sliceStridedSlice1model_2/model_1/conv2d_transpose_1/Shape:output:0?model_2/model_1/conv2d_transpose_1/strided_slice/stack:output:0Amodel_2/model_1/conv2d_transpose_1/strided_slice/stack_1:output:0Amodel_2/model_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*model_2/model_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :pl
*model_2/model_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :pl
*model_2/model_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
(model_2/model_1/conv2d_transpose_1/stackPack9model_2/model_1/conv2d_transpose_1/strided_slice:output:03model_2/model_1/conv2d_transpose_1/stack/1:output:03model_2/model_1/conv2d_transpose_1/stack/2:output:03model_2/model_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:�
8model_2/model_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
:model_2/model_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
:model_2/model_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
2model_2/model_1/conv2d_transpose_1/strided_slice_1StridedSlice1model_2/model_1/conv2d_transpose_1/stack:output:0Amodel_2/model_1/conv2d_transpose_1/strided_slice_1/stack:output:0Cmodel_2/model_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0Cmodel_2/model_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Bmodel_2/model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpKmodel_2_model_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
3model_2/model_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput1model_2/model_1/conv2d_transpose_1/stack:output:0Jmodel_2/model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:03model_2/model_1/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
9model_2/model_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
*model_2/model_1/conv2d_transpose_1/BiasAddBiasAdd<model_2/model_1/conv2d_transpose_1/conv2d_transpose:output:0Amodel_2/model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp�
'model_2/model_1/conv2d_transpose_1/ReluRelu3model_2/model_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp�
(model_2/model_1/conv2d_transpose_2/ShapeShape5model_2/model_1/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:�
6model_2/model_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
8model_2/model_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
8model_2/model_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0model_2/model_1/conv2d_transpose_2/strided_sliceStridedSlice1model_2/model_1/conv2d_transpose_2/Shape:output:0?model_2/model_1/conv2d_transpose_2/strided_slice/stack:output:0Amodel_2/model_1/conv2d_transpose_2/strided_slice/stack_1:output:0Amodel_2/model_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
*model_2/model_1/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�m
*model_2/model_1/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�l
*model_2/model_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
(model_2/model_1/conv2d_transpose_2/stackPack9model_2/model_1/conv2d_transpose_2/strided_slice:output:03model_2/model_1/conv2d_transpose_2/stack/1:output:03model_2/model_1/conv2d_transpose_2/stack/2:output:03model_2/model_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:�
8model_2/model_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
:model_2/model_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
:model_2/model_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
2model_2/model_1/conv2d_transpose_2/strided_slice_1StridedSlice1model_2/model_1/conv2d_transpose_2/stack:output:0Amodel_2/model_1/conv2d_transpose_2/strided_slice_1/stack:output:0Cmodel_2/model_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0Cmodel_2/model_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Bmodel_2/model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpKmodel_2_model_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0�
3model_2/model_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput1model_2/model_1/conv2d_transpose_2/stack:output:0Jmodel_2/model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:05model_2/model_1/conv2d_transpose_1/Relu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
9model_2/model_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
*model_2/model_1/conv2d_transpose_2/BiasAddBiasAdd<model_2/model_1/conv2d_transpose_2/conv2d_transpose:output:0Amodel_2/model_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
'model_2/model_1/conv2d_transpose_2/ReluRelu3model_2/model_1/conv2d_transpose_2/BiasAdd:output:0*
T0*1
_output_shapes
:������������
(model_2/model_1/conv2d_transpose_3/ShapeShape5model_2/model_1/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:�
6model_2/model_1/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
8model_2/model_1/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
8model_2/model_1/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0model_2/model_1/conv2d_transpose_3/strided_sliceStridedSlice1model_2/model_1/conv2d_transpose_3/Shape:output:0?model_2/model_1/conv2d_transpose_3/strided_slice/stack:output:0Amodel_2/model_1/conv2d_transpose_3/strided_slice/stack_1:output:0Amodel_2/model_1/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
*model_2/model_1/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�m
*model_2/model_1/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�l
*model_2/model_1/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
(model_2/model_1/conv2d_transpose_3/stackPack9model_2/model_1/conv2d_transpose_3/strided_slice:output:03model_2/model_1/conv2d_transpose_3/stack/1:output:03model_2/model_1/conv2d_transpose_3/stack/2:output:03model_2/model_1/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:�
8model_2/model_1/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
:model_2/model_1/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
:model_2/model_1/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
2model_2/model_1/conv2d_transpose_3/strided_slice_1StridedSlice1model_2/model_1/conv2d_transpose_3/stack:output:0Amodel_2/model_1/conv2d_transpose_3/strided_slice_1/stack:output:0Cmodel_2/model_1/conv2d_transpose_3/strided_slice_1/stack_1:output:0Cmodel_2/model_1/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Bmodel_2/model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpKmodel_2_model_1_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0�
3model_2/model_1/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput1model_2/model_1/conv2d_transpose_3/stack:output:0Jmodel_2/model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:05model_2/model_1/conv2d_transpose_2/Relu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
9model_2/model_1/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
*model_2/model_1/conv2d_transpose_3/BiasAddBiasAdd<model_2/model_1/conv2d_transpose_3/conv2d_transpose:output:0Amodel_2/model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
*model_2/model_1/conv2d_transpose_3/SigmoidSigmoid3model_2/model_1/conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:������������
Emodel_2/tf.keras.backend.binary_crossentropy/logistic_loss/zeros_like	ZerosLike3model_2/model_1/conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:������������
Gmodel_2/tf.keras.backend.binary_crossentropy/logistic_loss/GreaterEqualGreaterEqual3model_2/model_1/conv2d_transpose_3/BiasAdd:output:0Imodel_2/tf.keras.backend.binary_crossentropy/logistic_loss/zeros_like:y:0*
T0*1
_output_shapes
:������������
Amodel_2/tf.keras.backend.binary_crossentropy/logistic_loss/SelectSelectKmodel_2/tf.keras.backend.binary_crossentropy/logistic_loss/GreaterEqual:z:03model_2/model_1/conv2d_transpose_3/BiasAdd:output:0Imodel_2/tf.keras.backend.binary_crossentropy/logistic_loss/zeros_like:y:0*
T0*1
_output_shapes
:������������
>model_2/tf.keras.backend.binary_crossentropy/logistic_loss/NegNeg3model_2/model_1/conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:������������
Cmodel_2/tf.keras.backend.binary_crossentropy/logistic_loss/Select_1SelectKmodel_2/tf.keras.backend.binary_crossentropy/logistic_loss/GreaterEqual:z:0Bmodel_2/tf.keras.backend.binary_crossentropy/logistic_loss/Neg:y:03model_2/model_1/conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:������������
>model_2/tf.keras.backend.binary_crossentropy/logistic_loss/mulMul3model_2/model_1/conv2d_transpose_3/BiasAdd:output:0input_1*
T0*1
_output_shapes
:������������
>model_2/tf.keras.backend.binary_crossentropy/logistic_loss/subSubJmodel_2/tf.keras.backend.binary_crossentropy/logistic_loss/Select:output:0Bmodel_2/tf.keras.backend.binary_crossentropy/logistic_loss/mul:z:0*
T0*1
_output_shapes
:������������
>model_2/tf.keras.backend.binary_crossentropy/logistic_loss/ExpExpLmodel_2/tf.keras.backend.binary_crossentropy/logistic_loss/Select_1:output:0*
T0*1
_output_shapes
:������������
@model_2/tf.keras.backend.binary_crossentropy/logistic_loss/Log1pLog1pBmodel_2/tf.keras.backend.binary_crossentropy/logistic_loss/Exp:y:0*
T0*1
_output_shapes
:������������
:model_2/tf.keras.backend.binary_crossentropy/logistic_lossAddV2Bmodel_2/tf.keras.backend.binary_crossentropy/logistic_loss/sub:z:0Dmodel_2/tf.keras.backend.binary_crossentropy/logistic_loss/Log1p:y:0*
T0*1
_output_shapes
:�����������z
model_2/tf.math.exp_1/ExpExp&model_2/model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
$model_2/tf.__operators__.add_1/AddV2AddV2model_2_238681&model_2/model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@
model_2/tf.math.square/SquareSquare$model_2/model/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
0model_2/tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         �
model_2/tf.math.reduce_sum/SumSum>model_2/tf.keras.backend.binary_crossentropy/logistic_loss:z:09model_2/tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:����������
model_2/tf.math.subtract/SubSub(model_2/tf.__operators__.add_1/AddV2:z:0!model_2/tf.math.square/Square:y:0*
T0*'
_output_shapes
:���������@�
model_2/tf.math.subtract_1/SubSub model_2/tf.math.subtract/Sub:z:0model_2/tf.math.exp_1/Exp:y:0*
T0*'
_output_shapes
:���������@m
#model_2/tf.math.reduce_mean_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
"model_2/tf.math.reduce_mean_1/MeanMean'model_2/tf.math.reduce_sum/Sum:output:0,model_2/tf.math.reduce_mean_1/Const:output:0*
T0*
_output_shapes
: }
2model_2/tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
 model_2/tf.math.reduce_sum_1/SumSum"model_2/tf.math.subtract_1/Sub:z:0;model_2/tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:���������Y
model_2/add_metric/RankConst*
_output_shapes
: *
dtype0*
value	B : `
model_2/add_metric/range/startConst*
_output_shapes
: *
dtype0*
value	B : `
model_2/add_metric/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
model_2/add_metric/rangeRange'model_2/add_metric/range/start:output:0 model_2/add_metric/Rank:output:0'model_2/add_metric/range/delta:output:0*
_output_shapes
: �
model_2/add_metric/SumSum+model_2/tf.math.reduce_mean_1/Mean:output:0!model_2/add_metric/range:output:0*
T0*
_output_shapes
: �
&model_2/add_metric/AssignAddVariableOpAssignAddVariableOp/model_2_add_metric_assignaddvariableop_resourcemodel_2/add_metric/Sum:output:0*
_output_shapes
 *
dtype0Y
model_2/add_metric/SizeConst*
_output_shapes
: *
dtype0*
value	B :q
model_2/add_metric/CastCast model_2/add_metric/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: �
(model_2/add_metric/AssignAddVariableOp_1AssignAddVariableOp1model_2_add_metric_assignaddvariableop_1_resourcemodel_2/add_metric/Cast:y:0'^model_2/add_metric/AssignAddVariableOp*
_output_shapes
 *
dtype0�
,model_2/add_metric/div_no_nan/ReadVariableOpReadVariableOp/model_2_add_metric_assignaddvariableop_resource'^model_2/add_metric/AssignAddVariableOp)^model_2/add_metric/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
.model_2/add_metric/div_no_nan/ReadVariableOp_1ReadVariableOp1model_2_add_metric_assignaddvariableop_1_resource)^model_2/add_metric/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
model_2/add_metric/div_no_nanDivNoNan4model_2/add_metric/div_no_nan/ReadVariableOp:value:06model_2/add_metric/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: k
model_2/add_metric/IdentityIdentity!model_2/add_metric/div_no_nan:z:0*
T0*
_output_shapes
: �
model_2/tf.math.multiply_1/MulMulmodel_2_238706)model_2/tf.math.reduce_sum_1/Sum:output:0*
T0*#
_output_shapes
:����������
$model_2/tf.__operators__.add_2/AddV2AddV2'model_2/tf.math.reduce_sum/Sum:output:0"model_2/tf.math.multiply_1/Mul:z:0*
T0*#
_output_shapes
:���������m
#model_2/tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
"model_2/tf.math.reduce_mean_2/MeanMean"model_2/tf.math.multiply_1/Mul:z:0,model_2/tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: k
!model_2/tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
 model_2/tf.math.reduce_mean/MeanMean(model_2/tf.__operators__.add_2/AddV2:z:0*model_2/tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: [
model_2/add_metric_1/RankConst*
_output_shapes
: *
dtype0*
value	B : b
 model_2/add_metric_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 model_2/add_metric_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
model_2/add_metric_1/rangeRange)model_2/add_metric_1/range/start:output:0"model_2/add_metric_1/Rank:output:0)model_2/add_metric_1/range/delta:output:0*
_output_shapes
: �
model_2/add_metric_1/SumSum+model_2/tf.math.reduce_mean_2/Mean:output:0#model_2/add_metric_1/range:output:0*
T0*
_output_shapes
: �
(model_2/add_metric_1/AssignAddVariableOpAssignAddVariableOp1model_2_add_metric_1_assignaddvariableop_resource!model_2/add_metric_1/Sum:output:0*
_output_shapes
 *
dtype0[
model_2/add_metric_1/SizeConst*
_output_shapes
: *
dtype0*
value	B :u
model_2/add_metric_1/CastCast"model_2/add_metric_1/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: �
*model_2/add_metric_1/AssignAddVariableOp_1AssignAddVariableOp3model_2_add_metric_1_assignaddvariableop_1_resourcemodel_2/add_metric_1/Cast:y:0)^model_2/add_metric_1/AssignAddVariableOp*
_output_shapes
 *
dtype0�
.model_2/add_metric_1/div_no_nan/ReadVariableOpReadVariableOp1model_2_add_metric_1_assignaddvariableop_resource)^model_2/add_metric_1/AssignAddVariableOp+^model_2/add_metric_1/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
0model_2/add_metric_1/div_no_nan/ReadVariableOp_1ReadVariableOp3model_2_add_metric_1_assignaddvariableop_1_resource+^model_2/add_metric_1/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
model_2/add_metric_1/div_no_nanDivNoNan6model_2/add_metric_1/div_no_nan/ReadVariableOp:value:08model_2/add_metric_1/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: o
model_2/add_metric_1/IdentityIdentity#model_2/add_metric_1/div_no_nan:z:0*
T0*
_output_shapes
: [
model_2/add_metric_2/RankConst*
_output_shapes
: *
dtype0*
value	B : b
 model_2/add_metric_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 model_2/add_metric_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
model_2/add_metric_2/rangeRange)model_2/add_metric_2/range/start:output:0"model_2/add_metric_2/Rank:output:0)model_2/add_metric_2/range/delta:output:0*
_output_shapes
: �
model_2/add_metric_2/SumSum)model_2/tf.math.reduce_mean/Mean:output:0#model_2/add_metric_2/range:output:0*
T0*
_output_shapes
: �
(model_2/add_metric_2/AssignAddVariableOpAssignAddVariableOp1model_2_add_metric_2_assignaddvariableop_resource!model_2/add_metric_2/Sum:output:0*
_output_shapes
 *
dtype0[
model_2/add_metric_2/SizeConst*
_output_shapes
: *
dtype0*
value	B :u
model_2/add_metric_2/CastCast"model_2/add_metric_2/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: �
*model_2/add_metric_2/AssignAddVariableOp_1AssignAddVariableOp3model_2_add_metric_2_assignaddvariableop_1_resourcemodel_2/add_metric_2/Cast:y:0)^model_2/add_metric_2/AssignAddVariableOp*
_output_shapes
 *
dtype0�
.model_2/add_metric_2/div_no_nan/ReadVariableOpReadVariableOp1model_2_add_metric_2_assignaddvariableop_resource)^model_2/add_metric_2/AssignAddVariableOp+^model_2/add_metric_2/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
0model_2/add_metric_2/div_no_nan/ReadVariableOp_1ReadVariableOp3model_2_add_metric_2_assignaddvariableop_1_resource+^model_2/add_metric_2/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
model_2/add_metric_2/div_no_nanDivNoNan6model_2/add_metric_2/div_no_nan/ReadVariableOp:value:08model_2/add_metric_2/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: o
model_2/add_metric_2/IdentityIdentity#model_2/add_metric_2/div_no_nan:z:0*
T0*
_output_shapes
: �
IdentityIdentity.model_2/model_1/conv2d_transpose_3/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp'^model_2/add_metric/AssignAddVariableOp)^model_2/add_metric/AssignAddVariableOp_1-^model_2/add_metric/div_no_nan/ReadVariableOp/^model_2/add_metric/div_no_nan/ReadVariableOp_1)^model_2/add_metric_1/AssignAddVariableOp+^model_2/add_metric_1/AssignAddVariableOp_1/^model_2/add_metric_1/div_no_nan/ReadVariableOp1^model_2/add_metric_1/div_no_nan/ReadVariableOp_1)^model_2/add_metric_2/AssignAddVariableOp+^model_2/add_metric_2/AssignAddVariableOp_1/^model_2/add_metric_2/div_no_nan/ReadVariableOp1^model_2/add_metric_2/div_no_nan/ReadVariableOp_1,^model_2/model/conv2d/BiasAdd/ReadVariableOp+^model_2/model/conv2d/Conv2D/ReadVariableOp.^model_2/model/conv2d_1/BiasAdd/ReadVariableOp-^model_2/model/conv2d_1/Conv2D/ReadVariableOp.^model_2/model/conv2d_2/BiasAdd/ReadVariableOp-^model_2/model/conv2d_2/Conv2D/ReadVariableOp+^model_2/model/dense/BiasAdd/ReadVariableOp*^model_2/model/dense/MatMul/ReadVariableOp-^model_2/model/dense_1/BiasAdd/ReadVariableOp,^model_2/model/dense_1/MatMul/ReadVariableOp8^model_2/model_1/conv2d_transpose/BiasAdd/ReadVariableOpA^model_2/model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:^model_2/model_1/conv2d_transpose_1/BiasAdd/ReadVariableOpC^model_2/model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:^model_2/model_1/conv2d_transpose_2/BiasAdd/ReadVariableOpC^model_2/model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:^model_2/model_1/conv2d_transpose_3/BiasAdd/ReadVariableOpC^model_2/model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp/^model_2/model_1/dense_2/BiasAdd/ReadVariableOp.^model_2/model_1/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&model_2/add_metric/AssignAddVariableOp&model_2/add_metric/AssignAddVariableOp2T
(model_2/add_metric/AssignAddVariableOp_1(model_2/add_metric/AssignAddVariableOp_12\
,model_2/add_metric/div_no_nan/ReadVariableOp,model_2/add_metric/div_no_nan/ReadVariableOp2`
.model_2/add_metric/div_no_nan/ReadVariableOp_1.model_2/add_metric/div_no_nan/ReadVariableOp_12T
(model_2/add_metric_1/AssignAddVariableOp(model_2/add_metric_1/AssignAddVariableOp2X
*model_2/add_metric_1/AssignAddVariableOp_1*model_2/add_metric_1/AssignAddVariableOp_12`
.model_2/add_metric_1/div_no_nan/ReadVariableOp.model_2/add_metric_1/div_no_nan/ReadVariableOp2d
0model_2/add_metric_1/div_no_nan/ReadVariableOp_10model_2/add_metric_1/div_no_nan/ReadVariableOp_12T
(model_2/add_metric_2/AssignAddVariableOp(model_2/add_metric_2/AssignAddVariableOp2X
*model_2/add_metric_2/AssignAddVariableOp_1*model_2/add_metric_2/AssignAddVariableOp_12`
.model_2/add_metric_2/div_no_nan/ReadVariableOp.model_2/add_metric_2/div_no_nan/ReadVariableOp2d
0model_2/add_metric_2/div_no_nan/ReadVariableOp_10model_2/add_metric_2/div_no_nan/ReadVariableOp_12Z
+model_2/model/conv2d/BiasAdd/ReadVariableOp+model_2/model/conv2d/BiasAdd/ReadVariableOp2X
*model_2/model/conv2d/Conv2D/ReadVariableOp*model_2/model/conv2d/Conv2D/ReadVariableOp2^
-model_2/model/conv2d_1/BiasAdd/ReadVariableOp-model_2/model/conv2d_1/BiasAdd/ReadVariableOp2\
,model_2/model/conv2d_1/Conv2D/ReadVariableOp,model_2/model/conv2d_1/Conv2D/ReadVariableOp2^
-model_2/model/conv2d_2/BiasAdd/ReadVariableOp-model_2/model/conv2d_2/BiasAdd/ReadVariableOp2\
,model_2/model/conv2d_2/Conv2D/ReadVariableOp,model_2/model/conv2d_2/Conv2D/ReadVariableOp2X
*model_2/model/dense/BiasAdd/ReadVariableOp*model_2/model/dense/BiasAdd/ReadVariableOp2V
)model_2/model/dense/MatMul/ReadVariableOp)model_2/model/dense/MatMul/ReadVariableOp2\
,model_2/model/dense_1/BiasAdd/ReadVariableOp,model_2/model/dense_1/BiasAdd/ReadVariableOp2Z
+model_2/model/dense_1/MatMul/ReadVariableOp+model_2/model/dense_1/MatMul/ReadVariableOp2r
7model_2/model_1/conv2d_transpose/BiasAdd/ReadVariableOp7model_2/model_1/conv2d_transpose/BiasAdd/ReadVariableOp2�
@model_2/model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp@model_2/model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2v
9model_2/model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp9model_2/model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2�
Bmodel_2/model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpBmodel_2/model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2v
9model_2/model_1/conv2d_transpose_2/BiasAdd/ReadVariableOp9model_2/model_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2�
Bmodel_2/model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpBmodel_2/model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2v
9model_2/model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp9model_2/model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp2�
Bmodel_2/model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOpBmodel_2/model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2`
.model_2/model_1/dense_2/BiasAdd/ReadVariableOp.model_2/model_1/dense_2/BiasAdd/ReadVariableOp2^
-model_2/model_1/dense_2/MatMul/ReadVariableOp-model_2/model_1/dense_2/MatMul/ReadVariableOp:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: 
�!
�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_241824

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
H__inference_add_metric_2_layer_call_and_return_conditional_losses_241504

inputs&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: 

identity_1��AssignAddVariableOp�AssignAddVariableOp_1�div_no_nan/ReadVariableOp�div_no_nan/ReadVariableOp_1F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: C
SumSuminputsrange:output:0*
T0*
_output_shapes
: y
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: �
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0�
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: F

Identity_1Identityinputs^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_126
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_1:> :

_output_shapes
: 
 
_user_specified_nameinputs
�
�
A__inference_model_layer_call_and_return_conditional_losses_238982

inputs'
conv2d_238954:
conv2d_238956:)
conv2d_1_238959:
conv2d_1_238961:)
conv2d_2_238964: 
conv2d_2_238966: "
dense_1_238970:
��@
dense_1_238972:@ 
dense_238975:
��@
dense_238977:@
identity

identity_1��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_238954conv2d_238956*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_238759�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_238959conv2d_1_238961*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_238776�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_238964conv2d_2_238966*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_238793�
flatten/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_238805�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_238970dense_1_238972*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_238817�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_238975dense_238977*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_238833u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@y

Identity_1Identity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
p
D__inference_add_loss_layer_call_and_return_conditional_losses_239701

inputs
identity

identity_1=
IdentityIdentityinputs*
T0*
_output_shapes
: ?

Identity_1Identityinputs*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
�
�
-__inference_add_metric_2_layer_call_fn_241487

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_add_metric_2_layer_call_and_return_conditional_losses_239721^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:> :

_output_shapes
: 
 
_user_specified_nameinputs
�/
�
A__inference_model_layer_call_and_return_conditional_losses_241163

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource: :
&dense_1_matmul_readvariableop_resource:
��@5
'dense_1_biasadd_readvariableop_resource:@8
$dense_matmul_readvariableop_resource:
��@3
%dense_biasadd_readvariableop_resource:@
identity

identity_1��conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ppf
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������88�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:��������� ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� b  �
flatten/ReshapeReshapeconv2d_2/Relu:activations:0flatten/Const:output:0*
T0*)
_output_shapes
:������������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��@*
dtype0�
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��@*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@i

Identity_1Identitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
(__inference_model_2_layer_call_fn_239789
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
��@
	unknown_6:@
	unknown_7:
��@
	unknown_8:@
	unknown_9:
@��

unknown_10:
��$

unknown_11:  

unknown_12: $

unknown_13: 

unknown_14:$

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19

unknown_20: 

unknown_21: 

unknown_22

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:�����������: *6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_239729y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: 
�	
�
C__inference_dense_1_layer_call_and_return_conditional_losses_238817

inputs2
matmul_readvariableop_resource:
��@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
F__inference_add_metric_layer_call_and_return_conditional_losses_239659

inputs&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: 

identity_1��AssignAddVariableOp�AssignAddVariableOp_1�div_no_nan/ReadVariableOp�div_no_nan/ReadVariableOp_1F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: C
SumSuminputsrange:output:0*
T0*
_output_shapes
: y
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: �
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0�
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: F

Identity_1Identityinputs^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_126
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_1:> :

_output_shapes
: 
 
_user_specified_nameinputs
�i
�	
C__inference_model_2_layer_call_and_return_conditional_losses_239729

inputs&
model_239548:
model_239550:&
model_239552:
model_239554:&
model_239556: 
model_239558:  
model_239560:
��@
model_239562:@ 
model_239564:
��@
model_239566:@"
model_1_239589:
@��
model_1_239591:
��(
model_1_239593:  
model_1_239595: (
model_1_239597: 
model_1_239599:(
model_1_239601:
model_1_239603:(
model_1_239605:
model_1_239607:
unknown
add_metric_239660: 
add_metric_239662: 
	unknown_0
add_metric_1_239691: 
add_metric_1_239693: 
add_metric_2_239722: 
add_metric_2_239724: 
identity

identity_1��"add_metric/StatefulPartitionedCall�$add_metric_1/StatefulPartitionedCall�$add_metric_2/StatefulPartitionedCall�model/StatefulPartitionedCall�model_1/StatefulPartitionedCall�
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_239548model_239550model_239552model_239554model_239556model_239558model_239560model_239562model_239564model_239566*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_238841p
tf.compat.v1.shape_1/ShapeShape&model/StatefulPartitionedCall:output:1*
T0*
_output_shapes
:n
tf.compat.v1.shape/ShapeShape&model/StatefulPartitionedCall:output:1*
T0*
_output_shapes
:v
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&tf.__operators__.getitem/strided_sliceStridedSlice!tf.compat.v1.shape/Shape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(tf.__operators__.getitem_1/strided_sliceStridedSlice#tf.compat.v1.shape_1/Shape:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
tf.math.exp/ExpExp&model/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:���������@�
$tf.random.normal/random_normal/shapePack/tf.__operators__.getitem/strided_slice:output:01tf.__operators__.getitem_1/strided_slice:output:0*
N*
T0*
_output_shapes
:h
#tf.random.normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%tf.random.normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
3tf.random.normal/random_normal/RandomStandardNormalRandomStandardNormal-tf.random.normal/random_normal/shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0�
"tf.random.normal/random_normal/mulMul<tf.random.normal/random_normal/RandomStandardNormal:output:0.tf.random.normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:���������@�
tf.random.normal/random_normalAddV2&tf.random.normal/random_normal/mul:z:0,tf.random.normal/random_normal/mean:output:0*
T0*'
_output_shapes
:���������@�
tf.math.multiply/MulMultf.math.exp/Exp:y:0"tf.random.normal/random_normal:z:0*
T0*'
_output_shapes
:���������@�
tf.__operators__.add/AddV2AddV2&model/StatefulPartitionedCall:output:0tf.math.multiply/Mul:z:0*
T0*'
_output_shapes
:���������@�
model_1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0model_1_239589model_1_239591model_1_239593model_1_239595model_1_239597model_1_239599model_1_239601model_1_239603model_1_239605model_1_239607*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_239337o
*tf.keras.backend.binary_crossentropy/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���3o
*tf.keras.backend.binary_crossentropy/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
(tf.keras.backend.binary_crossentropy/subSub3tf.keras.backend.binary_crossentropy/sub/x:output:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*
_output_shapes
: �
:tf.keras.backend.binary_crossentropy/clip_by_value/MinimumMinimum(model_1/StatefulPartitionedCall:output:0,tf.keras.backend.binary_crossentropy/sub:z:0*
T0*1
_output_shapes
:������������
2tf.keras.backend.binary_crossentropy/clip_by_valueMaximum>tf.keras.backend.binary_crossentropy/clip_by_value/Minimum:z:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*1
_output_shapes
:�����������o
*tf.keras.backend.binary_crossentropy/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
(tf.keras.backend.binary_crossentropy/addAddV26tf.keras.backend.binary_crossentropy/clip_by_value:z:03tf.keras.backend.binary_crossentropy/add/y:output:0*
T0*1
_output_shapes
:������������
(tf.keras.backend.binary_crossentropy/LogLog,tf.keras.backend.binary_crossentropy/add:z:0*
T0*1
_output_shapes
:������������
(tf.keras.backend.binary_crossentropy/mulMulinputs,tf.keras.backend.binary_crossentropy/Log:y:0*
T0*1
_output_shapes
:�����������q
,tf.keras.backend.binary_crossentropy/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
*tf.keras.backend.binary_crossentropy/sub_1Sub5tf.keras.backend.binary_crossentropy/sub_1/x:output:0inputs*
T0*1
_output_shapes
:�����������q
,tf.keras.backend.binary_crossentropy/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
*tf.keras.backend.binary_crossentropy/sub_2Sub5tf.keras.backend.binary_crossentropy/sub_2/x:output:06tf.keras.backend.binary_crossentropy/clip_by_value:z:0*
T0*1
_output_shapes
:�����������q
,tf.keras.backend.binary_crossentropy/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
*tf.keras.backend.binary_crossentropy/add_1AddV2.tf.keras.backend.binary_crossentropy/sub_2:z:05tf.keras.backend.binary_crossentropy/add_1/y:output:0*
T0*1
_output_shapes
:������������
*tf.keras.backend.binary_crossentropy/Log_1Log.tf.keras.backend.binary_crossentropy/add_1:z:0*
T0*1
_output_shapes
:������������
*tf.keras.backend.binary_crossentropy/mul_1Mul.tf.keras.backend.binary_crossentropy/sub_1:z:0.tf.keras.backend.binary_crossentropy/Log_1:y:0*
T0*1
_output_shapes
:������������
*tf.keras.backend.binary_crossentropy/add_2AddV2,tf.keras.backend.binary_crossentropy/mul:z:0.tf.keras.backend.binary_crossentropy/mul_1:z:0*
T0*1
_output_shapes
:������������
(tf.keras.backend.binary_crossentropy/NegNeg.tf.keras.backend.binary_crossentropy/add_2:z:0*
T0*1
_output_shapes
:�����������r
tf.math.exp_1/ExpExp&model/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:���������@�
tf.__operators__.add_1/AddV2AddV2unknown&model/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:���������@y
tf.math.square/SquareSquare&model/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@}
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         �
tf.math.reduce_sum/SumSum,tf.keras.backend.binary_crossentropy/Neg:y:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:����������
tf.math.subtract/SubSub tf.__operators__.add_1/AddV2:z:0tf.math.square/Square:y:0*
T0*'
_output_shapes
:���������@�
tf.math.subtract_1/SubSubtf.math.subtract/Sub:z:0tf.math.exp_1/Exp:y:0*
T0*'
_output_shapes
:���������@e
tf.math.reduce_mean_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_mean_1/MeanMeantf.math.reduce_sum/Sum:output:0$tf.math.reduce_mean_1/Const:output:0*
T0*
_output_shapes
: u
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
tf.math.reduce_sum_1/SumSumtf.math.subtract_1/Sub:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:����������
"add_metric/StatefulPartitionedCallStatefulPartitionedCall#tf.math.reduce_mean_1/Mean:output:0add_metric_239660add_metric_239662*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_add_metric_layer_call_and_return_conditional_losses_239659y
tf.math.multiply_1/MulMul	unknown_0!tf.math.reduce_sum_1/Sum:output:0*
T0*#
_output_shapes
:����������
tf.__operators__.add_2/AddV2AddV2tf.math.reduce_sum/Sum:output:0tf.math.multiply_1/Mul:z:0*
T0*#
_output_shapes
:���������e
tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_mean_2/MeanMeantf.math.multiply_1/Mul:z:0$tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: c
tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_mean/MeanMean tf.__operators__.add_2/AddV2:z:0"tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: �
$add_metric_1/StatefulPartitionedCallStatefulPartitionedCall#tf.math.reduce_mean_2/Mean:output:0add_metric_1_239691add_metric_1_239693*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_add_metric_1_layer_call_and_return_conditional_losses_239690�
add_loss/PartitionedCallPartitionedCall!tf.math.reduce_mean/Mean:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_add_loss_layer_call_and_return_conditional_losses_239701�
$add_metric_2/StatefulPartitionedCallStatefulPartitionedCall!tf.math.reduce_mean/Mean:output:0add_metric_2_239722add_metric_2_239724*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_add_metric_2_layer_call_and_return_conditional_losses_239721�
IdentityIdentity(model_1/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������a

Identity_1Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: �
NoOpNoOp#^add_metric/StatefulPartitionedCall%^add_metric_1/StatefulPartitionedCall%^add_metric_2/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"add_metric/StatefulPartitionedCall"add_metric/StatefulPartitionedCall2L
$add_metric_1/StatefulPartitionedCall$add_metric_1/StatefulPartitionedCall2L
$add_metric_2/StatefulPartitionedCall$add_metric_2/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�!
�
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_239134

inputsB
(conv2d_transpose_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
��
�
C__inference_model_2_layer_call_and_return_conditional_losses_241029

inputsE
+model_conv2d_conv2d_readvariableop_resource::
,model_conv2d_biasadd_readvariableop_resource:G
-model_conv2d_1_conv2d_readvariableop_resource:<
.model_conv2d_1_biasadd_readvariableop_resource:G
-model_conv2d_2_conv2d_readvariableop_resource: <
.model_conv2d_2_biasadd_readvariableop_resource: @
,model_dense_1_matmul_readvariableop_resource:
��@;
-model_dense_1_biasadd_readvariableop_resource:@>
*model_dense_matmul_readvariableop_resource:
��@9
+model_dense_biasadd_readvariableop_resource:@B
.model_1_dense_2_matmul_readvariableop_resource:
@��?
/model_1_dense_2_biasadd_readvariableop_resource:
��[
Amodel_1_conv2d_transpose_conv2d_transpose_readvariableop_resource:  F
8model_1_conv2d_transpose_biasadd_readvariableop_resource: ]
Cmodel_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource: H
:model_1_conv2d_transpose_1_biasadd_readvariableop_resource:]
Cmodel_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:H
:model_1_conv2d_transpose_2_biasadd_readvariableop_resource:]
Cmodel_1_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:H
:model_1_conv2d_transpose_3_biasadd_readvariableop_resource:
unknown1
'add_metric_assignaddvariableop_resource: 3
)add_metric_assignaddvariableop_1_resource: 
	unknown_03
)add_metric_1_assignaddvariableop_resource: 5
+add_metric_1_assignaddvariableop_1_resource: 3
)add_metric_2_assignaddvariableop_resource: 5
+add_metric_2_assignaddvariableop_1_resource: 
identity

identity_1��add_metric/AssignAddVariableOp� add_metric/AssignAddVariableOp_1�$add_metric/div_no_nan/ReadVariableOp�&add_metric/div_no_nan/ReadVariableOp_1� add_metric_1/AssignAddVariableOp�"add_metric_1/AssignAddVariableOp_1�&add_metric_1/div_no_nan/ReadVariableOp�(add_metric_1/div_no_nan/ReadVariableOp_1� add_metric_2/AssignAddVariableOp�"add_metric_2/AssignAddVariableOp_1�&add_metric_2/div_no_nan/ReadVariableOp�(add_metric_2/div_no_nan/ReadVariableOp_1�#model/conv2d/BiasAdd/ReadVariableOp�"model/conv2d/Conv2D/ReadVariableOp�%model/conv2d_1/BiasAdd/ReadVariableOp�$model/conv2d_1/Conv2D/ReadVariableOp�%model/conv2d_2/BiasAdd/ReadVariableOp�$model/conv2d_2/Conv2D/ReadVariableOp�"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�/model_1/conv2d_transpose/BiasAdd/ReadVariableOp�8model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp�1model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp�:model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp�1model_1/conv2d_transpose_2/BiasAdd/ReadVariableOp�:model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp�1model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp�:model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp�&model_1/dense_2/BiasAdd/ReadVariableOp�%model_1/dense_2/MatMul/ReadVariableOp�
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
model/conv2d/Conv2DConv2Dinputs*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ppr
model/conv2d/ReluRelumodel/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp�
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
model/conv2d_1/Conv2DConv2Dmodel/conv2d/Relu:activations:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88v
model/conv2d_1/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������88�
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
model/conv2d_2/Conv2DConv2D!model/conv2d_1/Relu:activations:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� v
model/conv2d_2/ReluRelumodel/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:��������� d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� b  �
model/flatten/ReshapeReshape!model/conv2d_2/Relu:activations:0model/flatten/Const:output:0*
T0*)
_output_shapes
:������������
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��@*
dtype0�
model/dense_1/MatMulMatMulmodel/flatten/Reshape:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource* 
_output_shapes
:
��@*
dtype0�
model/dense/MatMulMatMulmodel/flatten/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@h
tf.compat.v1.shape_1/ShapeShapemodel/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:f
tf.compat.v1.shape/ShapeShapemodel/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:v
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&tf.__operators__.getitem/strided_sliceStridedSlice!tf.compat.v1.shape/Shape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(tf.__operators__.getitem_1/strided_sliceStridedSlice#tf.compat.v1.shape_1/Shape:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
tf.math.exp/ExpExpmodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
$tf.random.normal/random_normal/shapePack/tf.__operators__.getitem/strided_slice:output:01tf.__operators__.getitem_1/strided_slice:output:0*
N*
T0*
_output_shapes
:h
#tf.random.normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%tf.random.normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
3tf.random.normal/random_normal/RandomStandardNormalRandomStandardNormal-tf.random.normal/random_normal/shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0�
"tf.random.normal/random_normal/mulMul<tf.random.normal/random_normal/RandomStandardNormal:output:0.tf.random.normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:���������@�
tf.random.normal/random_normalAddV2&tf.random.normal/random_normal/mul:z:0,tf.random.normal/random_normal/mean:output:0*
T0*'
_output_shapes
:���������@�
tf.math.multiply/MulMultf.math.exp/Exp:y:0"tf.random.normal/random_normal:z:0*
T0*'
_output_shapes
:���������@�
tf.__operators__.add/AddV2AddV2model/dense/BiasAdd:output:0tf.math.multiply/Mul:z:0*
T0*'
_output_shapes
:���������@�
%model_1/dense_2/MatMul/ReadVariableOpReadVariableOp.model_1_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
@��*
dtype0�
model_1/dense_2/MatMulMatMultf.__operators__.add/AddV2:z:0-model_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:������������
&model_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_2_biasadd_readvariableop_resource*
_output_shapes

:��*
dtype0�
model_1/dense_2/BiasAddBiasAdd model_1/dense_2/MatMul:product:0.model_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������r
model_1/dense_2/ReluRelu model_1/dense_2/BiasAdd:output:0*
T0*)
_output_shapes
:�����������g
model_1/reshape/ShapeShape"model_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:m
#model_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model_1/reshape/strided_sliceStridedSlicemodel_1/reshape/Shape:output:0,model_1/reshape/strided_slice/stack:output:0.model_1/reshape/strided_slice/stack_1:output:0.model_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :a
model_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :a
model_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : �
model_1/reshape/Reshape/shapePack&model_1/reshape/strided_slice:output:0(model_1/reshape/Reshape/shape/1:output:0(model_1/reshape/Reshape/shape/2:output:0(model_1/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
model_1/reshape/ReshapeReshape"model_1/dense_2/Relu:activations:0&model_1/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� n
model_1/conv2d_transpose/ShapeShape model_1/reshape/Reshape:output:0*
T0*
_output_shapes
:v
,model_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&model_1/conv2d_transpose/strided_sliceStridedSlice'model_1/conv2d_transpose/Shape:output:05model_1/conv2d_transpose/strided_slice/stack:output:07model_1/conv2d_transpose/strided_slice/stack_1:output:07model_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 model_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :8b
 model_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :8b
 model_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
model_1/conv2d_transpose/stackPack/model_1/conv2d_transpose/strided_slice:output:0)model_1/conv2d_transpose/stack/1:output:0)model_1/conv2d_transpose/stack/2:output:0)model_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:x
.model_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(model_1/conv2d_transpose/strided_slice_1StridedSlice'model_1/conv2d_transpose/stack:output:07model_1/conv2d_transpose/strided_slice_1/stack:output:09model_1/conv2d_transpose/strided_slice_1/stack_1:output:09model_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
8model_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0�
)model_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput'model_1/conv2d_transpose/stack:output:0@model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0 model_1/reshape/Reshape:output:0*
T0*/
_output_shapes
:���������88 *
paddingSAME*
strides
�
/model_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp8model_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
 model_1/conv2d_transpose/BiasAddBiasAdd2model_1/conv2d_transpose/conv2d_transpose:output:07model_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88 �
model_1/conv2d_transpose/ReluRelu)model_1/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:���������88 {
 model_1/conv2d_transpose_1/ShapeShape+model_1/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:x
.model_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(model_1/conv2d_transpose_1/strided_sliceStridedSlice)model_1/conv2d_transpose_1/Shape:output:07model_1/conv2d_transpose_1/strided_slice/stack:output:09model_1/conv2d_transpose_1/strided_slice/stack_1:output:09model_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"model_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :pd
"model_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :pd
"model_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
 model_1/conv2d_transpose_1/stackPack1model_1/conv2d_transpose_1/strided_slice:output:0+model_1/conv2d_transpose_1/stack/1:output:0+model_1/conv2d_transpose_1/stack/2:output:0+model_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:z
0model_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2model_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2model_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*model_1/conv2d_transpose_1/strided_slice_1StridedSlice)model_1/conv2d_transpose_1/stack:output:09model_1/conv2d_transpose_1/strided_slice_1/stack:output:0;model_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
:model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
+model_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_1/stack:output:0Bmodel_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0+model_1/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
1model_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
"model_1/conv2d_transpose_1/BiasAddBiasAdd4model_1/conv2d_transpose_1/conv2d_transpose:output:09model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp�
model_1/conv2d_transpose_1/ReluRelu+model_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp}
 model_1/conv2d_transpose_2/ShapeShape-model_1/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:x
.model_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(model_1/conv2d_transpose_2/strided_sliceStridedSlice)model_1/conv2d_transpose_2/Shape:output:07model_1/conv2d_transpose_2/strided_slice/stack:output:09model_1/conv2d_transpose_2/strided_slice/stack_1:output:09model_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"model_1/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�e
"model_1/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�d
"model_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
 model_1/conv2d_transpose_2/stackPack1model_1/conv2d_transpose_2/strided_slice:output:0+model_1/conv2d_transpose_2/stack/1:output:0+model_1/conv2d_transpose_2/stack/2:output:0+model_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:z
0model_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2model_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2model_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*model_1/conv2d_transpose_2/strided_slice_1StridedSlice)model_1/conv2d_transpose_2/stack:output:09model_1/conv2d_transpose_2/strided_slice_1/stack:output:0;model_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
:model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0�
+model_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_2/stack:output:0Bmodel_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0-model_1/conv2d_transpose_1/Relu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
1model_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
"model_1/conv2d_transpose_2/BiasAddBiasAdd4model_1/conv2d_transpose_2/conv2d_transpose:output:09model_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
model_1/conv2d_transpose_2/ReluRelu+model_1/conv2d_transpose_2/BiasAdd:output:0*
T0*1
_output_shapes
:�����������}
 model_1/conv2d_transpose_3/ShapeShape-model_1/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:x
.model_1/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_1/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_1/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(model_1/conv2d_transpose_3/strided_sliceStridedSlice)model_1/conv2d_transpose_3/Shape:output:07model_1/conv2d_transpose_3/strided_slice/stack:output:09model_1/conv2d_transpose_3/strided_slice/stack_1:output:09model_1/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"model_1/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�e
"model_1/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�d
"model_1/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
 model_1/conv2d_transpose_3/stackPack1model_1/conv2d_transpose_3/strided_slice:output:0+model_1/conv2d_transpose_3/stack/1:output:0+model_1/conv2d_transpose_3/stack/2:output:0+model_1/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:z
0model_1/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2model_1/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2model_1/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*model_1/conv2d_transpose_3/strided_slice_1StridedSlice)model_1/conv2d_transpose_3/stack:output:09model_1/conv2d_transpose_3/strided_slice_1/stack:output:0;model_1/conv2d_transpose_3/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
:model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0�
+model_1/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_3/stack:output:0Bmodel_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0-model_1/conv2d_transpose_2/Relu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
1model_1/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
"model_1/conv2d_transpose_3/BiasAddBiasAdd4model_1/conv2d_transpose_3/conv2d_transpose:output:09model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
"model_1/conv2d_transpose_3/SigmoidSigmoid+model_1/conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:������������
=tf.keras.backend.binary_crossentropy/logistic_loss/zeros_like	ZerosLike+model_1/conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:������������
?tf.keras.backend.binary_crossentropy/logistic_loss/GreaterEqualGreaterEqual+model_1/conv2d_transpose_3/BiasAdd:output:0Atf.keras.backend.binary_crossentropy/logistic_loss/zeros_like:y:0*
T0*1
_output_shapes
:������������
9tf.keras.backend.binary_crossentropy/logistic_loss/SelectSelectCtf.keras.backend.binary_crossentropy/logistic_loss/GreaterEqual:z:0+model_1/conv2d_transpose_3/BiasAdd:output:0Atf.keras.backend.binary_crossentropy/logistic_loss/zeros_like:y:0*
T0*1
_output_shapes
:������������
6tf.keras.backend.binary_crossentropy/logistic_loss/NegNeg+model_1/conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:������������
;tf.keras.backend.binary_crossentropy/logistic_loss/Select_1SelectCtf.keras.backend.binary_crossentropy/logistic_loss/GreaterEqual:z:0:tf.keras.backend.binary_crossentropy/logistic_loss/Neg:y:0+model_1/conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:������������
6tf.keras.backend.binary_crossentropy/logistic_loss/mulMul+model_1/conv2d_transpose_3/BiasAdd:output:0inputs*
T0*1
_output_shapes
:������������
6tf.keras.backend.binary_crossentropy/logistic_loss/subSubBtf.keras.backend.binary_crossentropy/logistic_loss/Select:output:0:tf.keras.backend.binary_crossentropy/logistic_loss/mul:z:0*
T0*1
_output_shapes
:������������
6tf.keras.backend.binary_crossentropy/logistic_loss/ExpExpDtf.keras.backend.binary_crossentropy/logistic_loss/Select_1:output:0*
T0*1
_output_shapes
:������������
8tf.keras.backend.binary_crossentropy/logistic_loss/Log1pLog1p:tf.keras.backend.binary_crossentropy/logistic_loss/Exp:y:0*
T0*1
_output_shapes
:������������
2tf.keras.backend.binary_crossentropy/logistic_lossAddV2:tf.keras.backend.binary_crossentropy/logistic_loss/sub:z:0<tf.keras.backend.binary_crossentropy/logistic_loss/Log1p:y:0*
T0*1
_output_shapes
:�����������j
tf.math.exp_1/ExpExpmodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
tf.__operators__.add_1/AddV2AddV2unknownmodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@o
tf.math.square/SquareSquaremodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@}
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         �
tf.math.reduce_sum/SumSum6tf.keras.backend.binary_crossentropy/logistic_loss:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:����������
tf.math.subtract/SubSub tf.__operators__.add_1/AddV2:z:0tf.math.square/Square:y:0*
T0*'
_output_shapes
:���������@�
tf.math.subtract_1/SubSubtf.math.subtract/Sub:z:0tf.math.exp_1/Exp:y:0*
T0*'
_output_shapes
:���������@e
tf.math.reduce_mean_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_mean_1/MeanMeantf.math.reduce_sum/Sum:output:0$tf.math.reduce_mean_1/Const:output:0*
T0*
_output_shapes
: u
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
tf.math.reduce_sum_1/SumSumtf.math.subtract_1/Sub:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:���������Q
add_metric/RankConst*
_output_shapes
: *
dtype0*
value	B : X
add_metric/range/startConst*
_output_shapes
: *
dtype0*
value	B : X
add_metric/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
add_metric/rangeRangeadd_metric/range/start:output:0add_metric/Rank:output:0add_metric/range/delta:output:0*
_output_shapes
: v
add_metric/SumSum#tf.math.reduce_mean_1/Mean:output:0add_metric/range:output:0*
T0*
_output_shapes
: �
add_metric/AssignAddVariableOpAssignAddVariableOp'add_metric_assignaddvariableop_resourceadd_metric/Sum:output:0*
_output_shapes
 *
dtype0Q
add_metric/SizeConst*
_output_shapes
: *
dtype0*
value	B :a
add_metric/CastCastadd_metric/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: �
 add_metric/AssignAddVariableOp_1AssignAddVariableOp)add_metric_assignaddvariableop_1_resourceadd_metric/Cast:y:0^add_metric/AssignAddVariableOp*
_output_shapes
 *
dtype0�
$add_metric/div_no_nan/ReadVariableOpReadVariableOp'add_metric_assignaddvariableop_resource^add_metric/AssignAddVariableOp!^add_metric/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
&add_metric/div_no_nan/ReadVariableOp_1ReadVariableOp)add_metric_assignaddvariableop_1_resource!^add_metric/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
add_metric/div_no_nanDivNoNan,add_metric/div_no_nan/ReadVariableOp:value:0.add_metric/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: [
add_metric/IdentityIdentityadd_metric/div_no_nan:z:0*
T0*
_output_shapes
: y
tf.math.multiply_1/MulMul	unknown_0!tf.math.reduce_sum_1/Sum:output:0*
T0*#
_output_shapes
:����������
tf.__operators__.add_2/AddV2AddV2tf.math.reduce_sum/Sum:output:0tf.math.multiply_1/Mul:z:0*
T0*#
_output_shapes
:���������e
tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_mean_2/MeanMeantf.math.multiply_1/Mul:z:0$tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: c
tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_mean/MeanMean tf.__operators__.add_2/AddV2:z:0"tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: S
add_metric_1/RankConst*
_output_shapes
: *
dtype0*
value	B : Z
add_metric_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : Z
add_metric_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
add_metric_1/rangeRange!add_metric_1/range/start:output:0add_metric_1/Rank:output:0!add_metric_1/range/delta:output:0*
_output_shapes
: z
add_metric_1/SumSum#tf.math.reduce_mean_2/Mean:output:0add_metric_1/range:output:0*
T0*
_output_shapes
: �
 add_metric_1/AssignAddVariableOpAssignAddVariableOp)add_metric_1_assignaddvariableop_resourceadd_metric_1/Sum:output:0*
_output_shapes
 *
dtype0S
add_metric_1/SizeConst*
_output_shapes
: *
dtype0*
value	B :e
add_metric_1/CastCastadd_metric_1/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"add_metric_1/AssignAddVariableOp_1AssignAddVariableOp+add_metric_1_assignaddvariableop_1_resourceadd_metric_1/Cast:y:0!^add_metric_1/AssignAddVariableOp*
_output_shapes
 *
dtype0�
&add_metric_1/div_no_nan/ReadVariableOpReadVariableOp)add_metric_1_assignaddvariableop_resource!^add_metric_1/AssignAddVariableOp#^add_metric_1/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
(add_metric_1/div_no_nan/ReadVariableOp_1ReadVariableOp+add_metric_1_assignaddvariableop_1_resource#^add_metric_1/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
add_metric_1/div_no_nanDivNoNan.add_metric_1/div_no_nan/ReadVariableOp:value:00add_metric_1/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: _
add_metric_1/IdentityIdentityadd_metric_1/div_no_nan:z:0*
T0*
_output_shapes
: S
add_metric_2/RankConst*
_output_shapes
: *
dtype0*
value	B : Z
add_metric_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : Z
add_metric_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
add_metric_2/rangeRange!add_metric_2/range/start:output:0add_metric_2/Rank:output:0!add_metric_2/range/delta:output:0*
_output_shapes
: x
add_metric_2/SumSum!tf.math.reduce_mean/Mean:output:0add_metric_2/range:output:0*
T0*
_output_shapes
: �
 add_metric_2/AssignAddVariableOpAssignAddVariableOp)add_metric_2_assignaddvariableop_resourceadd_metric_2/Sum:output:0*
_output_shapes
 *
dtype0S
add_metric_2/SizeConst*
_output_shapes
: *
dtype0*
value	B :e
add_metric_2/CastCastadd_metric_2/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"add_metric_2/AssignAddVariableOp_1AssignAddVariableOp+add_metric_2_assignaddvariableop_1_resourceadd_metric_2/Cast:y:0!^add_metric_2/AssignAddVariableOp*
_output_shapes
 *
dtype0�
&add_metric_2/div_no_nan/ReadVariableOpReadVariableOp)add_metric_2_assignaddvariableop_resource!^add_metric_2/AssignAddVariableOp#^add_metric_2/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
(add_metric_2/div_no_nan/ReadVariableOp_1ReadVariableOp+add_metric_2_assignaddvariableop_1_resource#^add_metric_2/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
add_metric_2/div_no_nanDivNoNan.add_metric_2/div_no_nan/ReadVariableOp:value:00add_metric_2/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: _
add_metric_2/IdentityIdentityadd_metric_2/div_no_nan:z:0*
T0*
_output_shapes
: 
IdentityIdentity&model_1/conv2d_transpose_3/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:�����������a

Identity_1Identity!tf.math.reduce_mean/Mean:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^add_metric/AssignAddVariableOp!^add_metric/AssignAddVariableOp_1%^add_metric/div_no_nan/ReadVariableOp'^add_metric/div_no_nan/ReadVariableOp_1!^add_metric_1/AssignAddVariableOp#^add_metric_1/AssignAddVariableOp_1'^add_metric_1/div_no_nan/ReadVariableOp)^add_metric_1/div_no_nan/ReadVariableOp_1!^add_metric_2/AssignAddVariableOp#^add_metric_2/AssignAddVariableOp_1'^add_metric_2/div_no_nan/ReadVariableOp)^add_metric_2/div_no_nan/ReadVariableOp_1$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp0^model_1/conv2d_transpose/BiasAdd/ReadVariableOp9^model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2^model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2^model_1/conv2d_transpose_2/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2^model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp'^model_1/dense_2/BiasAdd/ReadVariableOp&^model_1/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
add_metric/AssignAddVariableOpadd_metric/AssignAddVariableOp2D
 add_metric/AssignAddVariableOp_1 add_metric/AssignAddVariableOp_12L
$add_metric/div_no_nan/ReadVariableOp$add_metric/div_no_nan/ReadVariableOp2P
&add_metric/div_no_nan/ReadVariableOp_1&add_metric/div_no_nan/ReadVariableOp_12D
 add_metric_1/AssignAddVariableOp add_metric_1/AssignAddVariableOp2H
"add_metric_1/AssignAddVariableOp_1"add_metric_1/AssignAddVariableOp_12P
&add_metric_1/div_no_nan/ReadVariableOp&add_metric_1/div_no_nan/ReadVariableOp2T
(add_metric_1/div_no_nan/ReadVariableOp_1(add_metric_1/div_no_nan/ReadVariableOp_12D
 add_metric_2/AssignAddVariableOp add_metric_2/AssignAddVariableOp2H
"add_metric_2/AssignAddVariableOp_1"add_metric_2/AssignAddVariableOp_12P
&add_metric_2/div_no_nan/ReadVariableOp&add_metric_2/div_no_nan/ReadVariableOp2T
(add_metric_2/div_no_nan/ReadVariableOp_1(add_metric_2/div_no_nan/ReadVariableOp_12J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2b
/model_1/conv2d_transpose/BiasAdd/ReadVariableOp/model_1/conv2d_transpose/BiasAdd/ReadVariableOp2t
8model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp8model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2f
1model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2f
1model_1/conv2d_transpose_2/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2f
1model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2P
&model_1/dense_2/BiasAdd/ReadVariableOp&model_1/dense_2/BiasAdd/ReadVariableOp2N
%model_1/dense_2/MatMul/ReadVariableOp%model_1/dense_2/MatMul/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�!
�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_239269

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
B__inference_conv2d_layer_call_and_return_conditional_losses_241524

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ppX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������ppi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������ppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
-__inference_add_metric_1_layer_call_fn_241461

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_add_metric_1_layer_call_and_return_conditional_losses_239690^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:> :

_output_shapes
: 
 
_user_specified_nameinputs
�
�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_238793

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������88
 
_user_specified_nameinputs
�
�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_241544

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������88i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������88w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������pp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������pp
 
_user_specified_nameinputs
�!
�
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_241738

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
(__inference_dense_1_layer_call_fn_241603

inputs
unknown:
��@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_238817o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
)__inference_conv2d_1_layer_call_fn_241533

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_238776w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������88`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������pp: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������pp
 
_user_specified_nameinputs
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_238805

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� b  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
D
(__inference_reshape_layer_call_fn_241638

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_239314h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:�����������:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�"
�
C__inference_model_1_layer_call_and_return_conditional_losses_239433

inputs"
dense_2_239406:
@��
dense_2_239408:
��1
conv2d_transpose_239412:  %
conv2d_transpose_239414: 3
conv2d_transpose_1_239417: '
conv2d_transpose_1_239419:3
conv2d_transpose_2_239422:'
conv2d_transpose_2_239424:3
conv2d_transpose_3_239427:'
conv2d_transpose_3_239429:
identity��(conv2d_transpose/StatefulPartitionedCall�*conv2d_transpose_1/StatefulPartitionedCall�*conv2d_transpose_2/StatefulPartitionedCall�*conv2d_transpose_3/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_239406dense_2_239408*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_239294�
reshape/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_239314�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_239412conv2d_transpose_239414*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_239134�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_239417conv2d_transpose_1_239419*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_239179�
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_239422conv2d_transpose_2_239424*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_239224�
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_239427conv2d_transpose_3_239429*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_239269�
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_model_2_layer_call_fn_240515

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
��@
	unknown_6:@
	unknown_7:
��@
	unknown_8:@
	unknown_9:
@��

unknown_10:
��$

unknown_11:  

unknown_12: $

unknown_13: 

unknown_14:$

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19

unknown_20: 

unknown_21: 

unknown_22

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:�����������: *6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_239729y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
(__inference_dense_2_layer_call_fn_241622

inputs
unknown:
@��
	unknown_0:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_239294q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
H__inference_add_metric_1_layer_call_and_return_conditional_losses_239690

inputs&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: 

identity_1��AssignAddVariableOp�AssignAddVariableOp_1�div_no_nan/ReadVariableOp�div_no_nan/ReadVariableOp_1F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: C
SumSuminputsrange:output:0*
T0*
_output_shapes
: y
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: �
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0�
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: F

Identity_1Identityinputs^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_126
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_1:> :

_output_shapes
: 
 
_user_specified_nameinputs
�
�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_241564

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������88
 
_user_specified_nameinputs
�
�
)__inference_conv2d_2_layer_call_fn_241553

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_238793w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������88: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������88
 
_user_specified_nameinputs
�

�
(__inference_model_1_layer_call_fn_241213

inputs
unknown:
@��
	unknown_0:
��#
	unknown_1:  
	unknown_2: #
	unknown_3: 
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_239433y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_model_2_layer_call_fn_240136
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
��@
	unknown_6:@
	unknown_7:
��@
	unknown_8:@
	unknown_9:
@��

unknown_10:
��$

unknown_11:  

unknown_12: $

unknown_13: 

unknown_14:$

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19

unknown_20: 

unknown_21: 

unknown_22

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:�����������: *6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_240014y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: 
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_241575

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� b  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
3__inference_conv2d_transpose_3_layer_call_fn_241790

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_239269�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
ݝ
�/
"__inference__traced_restore_242297
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:<
"assignvariableop_2_conv2d_1_kernel:.
 assignvariableop_3_conv2d_1_bias:<
"assignvariableop_4_conv2d_2_kernel: .
 assignvariableop_5_conv2d_2_bias: 3
assignvariableop_6_dense_kernel:
��@+
assignvariableop_7_dense_bias:@5
!assignvariableop_8_dense_1_kernel:
��@-
assignvariableop_9_dense_1_bias:@6
"assignvariableop_10_dense_2_kernel:
@��0
 assignvariableop_11_dense_2_bias:
��E
+assignvariableop_12_conv2d_transpose_kernel:  7
)assignvariableop_13_conv2d_transpose_bias: G
-assignvariableop_14_conv2d_transpose_1_kernel: 9
+assignvariableop_15_conv2d_transpose_1_bias:G
-assignvariableop_16_conv2d_transpose_2_kernel:9
+assignvariableop_17_conv2d_transpose_2_bias:G
-assignvariableop_18_conv2d_transpose_3_kernel:9
+assignvariableop_19_conv2d_transpose_3_bias:'
assignvariableop_20_adam_iter:	 )
assignvariableop_21_adam_beta_1: )
assignvariableop_22_adam_beta_2: (
assignvariableop_23_adam_decay: 0
&assignvariableop_24_adam_learning_rate: #
assignvariableop_25_total: #
assignvariableop_26_count: .
$assignvariableop_27_add_metric_total: .
$assignvariableop_28_add_metric_count: 0
&assignvariableop_29_add_metric_1_total: 0
&assignvariableop_30_add_metric_1_count: 0
&assignvariableop_31_add_metric_2_total: 0
&assignvariableop_32_add_metric_2_count: B
(assignvariableop_33_adam_conv2d_kernel_m:4
&assignvariableop_34_adam_conv2d_bias_m:D
*assignvariableop_35_adam_conv2d_1_kernel_m:6
(assignvariableop_36_adam_conv2d_1_bias_m:D
*assignvariableop_37_adam_conv2d_2_kernel_m: 6
(assignvariableop_38_adam_conv2d_2_bias_m: ;
'assignvariableop_39_adam_dense_kernel_m:
��@3
%assignvariableop_40_adam_dense_bias_m:@=
)assignvariableop_41_adam_dense_1_kernel_m:
��@5
'assignvariableop_42_adam_dense_1_bias_m:@=
)assignvariableop_43_adam_dense_2_kernel_m:
@��7
'assignvariableop_44_adam_dense_2_bias_m:
��L
2assignvariableop_45_adam_conv2d_transpose_kernel_m:  >
0assignvariableop_46_adam_conv2d_transpose_bias_m: N
4assignvariableop_47_adam_conv2d_transpose_1_kernel_m: @
2assignvariableop_48_adam_conv2d_transpose_1_bias_m:N
4assignvariableop_49_adam_conv2d_transpose_2_kernel_m:@
2assignvariableop_50_adam_conv2d_transpose_2_bias_m:N
4assignvariableop_51_adam_conv2d_transpose_3_kernel_m:@
2assignvariableop_52_adam_conv2d_transpose_3_bias_m:B
(assignvariableop_53_adam_conv2d_kernel_v:4
&assignvariableop_54_adam_conv2d_bias_v:D
*assignvariableop_55_adam_conv2d_1_kernel_v:6
(assignvariableop_56_adam_conv2d_1_bias_v:D
*assignvariableop_57_adam_conv2d_2_kernel_v: 6
(assignvariableop_58_adam_conv2d_2_bias_v: ;
'assignvariableop_59_adam_dense_kernel_v:
��@3
%assignvariableop_60_adam_dense_bias_v:@=
)assignvariableop_61_adam_dense_1_kernel_v:
��@5
'assignvariableop_62_adam_dense_1_bias_v:@=
)assignvariableop_63_adam_dense_2_kernel_v:
@��7
'assignvariableop_64_adam_dense_2_bias_v:
��L
2assignvariableop_65_adam_conv2d_transpose_kernel_v:  >
0assignvariableop_66_adam_conv2d_transpose_bias_v: N
4assignvariableop_67_adam_conv2d_transpose_1_kernel_v: @
2assignvariableop_68_adam_conv2d_transpose_1_bias_v:N
4assignvariableop_69_adam_conv2d_transpose_2_kernel_v:@
2assignvariableop_70_adam_conv2d_transpose_2_bias_v:N
4assignvariableop_71_adam_conv2d_transpose_3_kernel_v:@
2assignvariableop_72_adam_conv2d_transpose_3_bias_v:
identity_74��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_8�AssignVariableOp_9�"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*�!
value�!B�!JB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*�
value�B�JB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*X
dtypesN
L2J	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp+assignvariableop_12_conv2d_transpose_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp)assignvariableop_13_conv2d_transpose_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp-assignvariableop_14_conv2d_transpose_1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp+assignvariableop_15_conv2d_transpose_1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp-assignvariableop_16_conv2d_transpose_2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp+assignvariableop_17_conv2d_transpose_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp-assignvariableop_18_conv2d_transpose_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp+assignvariableop_19_conv2d_transpose_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_iterIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_beta_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_beta_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_learning_rateIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp$assignvariableop_27_add_metric_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp$assignvariableop_28_add_metric_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp&assignvariableop_29_add_metric_1_totalIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp&assignvariableop_30_add_metric_1_countIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp&assignvariableop_31_add_metric_2_totalIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp&assignvariableop_32_add_metric_2_countIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_conv2d_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_conv2d_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv2d_1_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv2d_1_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_2_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_2_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adam_dense_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp%assignvariableop_40_adam_dense_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_1_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_1_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_2_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_2_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp2assignvariableop_45_adam_conv2d_transpose_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp0assignvariableop_46_adam_conv2d_transpose_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp4assignvariableop_47_adam_conv2d_transpose_1_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp2assignvariableop_48_adam_conv2d_transpose_1_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp4assignvariableop_49_adam_conv2d_transpose_2_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp2assignvariableop_50_adam_conv2d_transpose_2_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp4assignvariableop_51_adam_conv2d_transpose_3_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp2assignvariableop_52_adam_conv2d_transpose_3_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_conv2d_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp&assignvariableop_54_adam_conv2d_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_conv2d_1_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_conv2d_1_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_conv2d_2_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_conv2d_2_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp'assignvariableop_59_adam_dense_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp%assignvariableop_60_adam_dense_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp)assignvariableop_61_adam_dense_1_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp'assignvariableop_62_adam_dense_1_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp)assignvariableop_63_adam_dense_2_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp'assignvariableop_64_adam_dense_2_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp2assignvariableop_65_adam_conv2d_transpose_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp0assignvariableop_66_adam_conv2d_transpose_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp4assignvariableop_67_adam_conv2d_transpose_1_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp2assignvariableop_68_adam_conv2d_transpose_1_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp4assignvariableop_69_adam_conv2d_transpose_2_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp2assignvariableop_70_adam_conv2d_transpose_2_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp4assignvariableop_71_adam_conv2d_transpose_3_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp2assignvariableop_72_adam_conv2d_transpose_3_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_73Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_74IdentityIdentity_73:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_74Identity_74:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
(__inference_model_2_layer_call_fn_240577

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
��@
	unknown_6:@
	unknown_7:
��@
	unknown_8:@
	unknown_9:
@��

unknown_10:
��$

unknown_11:  

unknown_12: $

unknown_13: 

unknown_14:$

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19

unknown_20: 

unknown_21: 

unknown_22

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:�����������: *6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_240014y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
A__inference_model_layer_call_and_return_conditional_losses_238841

inputs'
conv2d_238760:
conv2d_238762:)
conv2d_1_238777:
conv2d_1_238779:)
conv2d_2_238794: 
conv2d_2_238796: "
dense_1_238818:
��@
dense_1_238820:@ 
dense_238834:
��@
dense_238836:@
identity

identity_1��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_238760conv2d_238762*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_238759�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_238777conv2d_1_238779*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_238776�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_238794conv2d_2_238796*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_238793�
flatten/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_238805�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_238818dense_1_238820*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_238817�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_238834dense_238836*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_238833u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@y

Identity_1Identity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
F__inference_add_metric_layer_call_and_return_conditional_losses_241452

inputs&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: 

identity_1��AssignAddVariableOp�AssignAddVariableOp_1�div_no_nan/ReadVariableOp�div_no_nan/ReadVariableOp_1F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: C
SumSuminputsrange:output:0*
T0*
_output_shapes
: y
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: �
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0�
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: F

Identity_1Identityinputs^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_126
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_1:> :

_output_shapes
: 
 
_user_specified_nameinputs
�
�
1__inference_conv2d_transpose_layer_call_fn_241661

inputs!
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_239134�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
H__inference_add_metric_2_layer_call_and_return_conditional_losses_239721

inputs&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: 

identity_1��AssignAddVariableOp�AssignAddVariableOp_1�div_no_nan/ReadVariableOp�div_no_nan/ReadVariableOp_1F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: C
SumSuminputsrange:output:0*
T0*
_output_shapes
: y
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: �
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0�
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: F

Identity_1Identityinputs^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_126
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_1:> :

_output_shapes
: 
 
_user_specified_nameinputs
�

�
C__inference_dense_2_layer_call_and_return_conditional_losses_241633

inputs2
matmul_readvariableop_resource:
@��/
biasadd_readvariableop_resource:
��
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
@��*
dtype0k
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������t
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:��*
dtype0x
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������R
ReluReluBiasAdd:output:0*
T0*)
_output_shapes
:�����������c
IdentityIdentityRelu:activations:0^NoOp*
T0*)
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�"
�
C__inference_model_1_layer_call_and_return_conditional_losses_239511
input_2"
dense_2_239484:
@��
dense_2_239486:
��1
conv2d_transpose_239490:  %
conv2d_transpose_239492: 3
conv2d_transpose_1_239495: '
conv2d_transpose_1_239497:3
conv2d_transpose_2_239500:'
conv2d_transpose_2_239502:3
conv2d_transpose_3_239505:'
conv2d_transpose_3_239507:
identity��(conv2d_transpose/StatefulPartitionedCall�*conv2d_transpose_1/StatefulPartitionedCall�*conv2d_transpose_2/StatefulPartitionedCall�*conv2d_transpose_3/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_2_239484dense_2_239486*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_239294�
reshape/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_239314�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_239490conv2d_transpose_239492*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_239134�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_239495conv2d_transpose_1_239497*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_239179�
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_239500conv2d_transpose_2_239502*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_239224�
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_239505conv2d_transpose_3_239507*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_239269�
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
'
_output_shapes
:���������@
!
_user_specified_name	input_2
�
�
3__inference_conv2d_transpose_1_layer_call_fn_241704

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_239179�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_239034
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
��@
	unknown_6:@
	unknown_7:
��@
	unknown_8:@
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_238982o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
A__inference_model_layer_call_and_return_conditional_losses_239096
input_1'
conv2d_239068:
conv2d_239070:)
conv2d_1_239073:
conv2d_1_239075:)
conv2d_2_239078: 
conv2d_2_239080: "
dense_1_239084:
��@
dense_1_239086:@ 
dense_239089:
��@
dense_239091:@
identity

identity_1��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_239068conv2d_239070*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_238759�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_239073conv2d_1_239075*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_238776�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_239078conv2d_2_239080*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_238793�
flatten/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_238805�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_239084dense_1_239086*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_238817�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_239089dense_239091*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_238833u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@y

Identity_1Identity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
E
)__inference_add_loss_layer_call_fn_241421

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_add_loss_layer_call_and_return_conditional_losses_239701O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
�
�
+__inference_add_metric_layer_call_fn_241435

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_add_metric_layer_call_and_return_conditional_losses_239659^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:> :

_output_shapes
: 
 
_user_specified_nameinputs
�!
�
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_239179

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�	
�
A__inference_dense_layer_call_and_return_conditional_losses_241594

inputs2
matmul_readvariableop_resource:
��@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
p
D__inference_add_loss_layer_call_and_return_conditional_losses_241426

inputs
identity

identity_1=
IdentityIdentityinputs*
T0*
_output_shapes
: ?

Identity_1Identityinputs*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
E
input_1:
serving_default_input_1:0�����������E
model_1:
StatefulPartitionedCall:0�����������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-1
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_default_save_signature
$	optimizer
%loss
&
signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
layer-0
'layer_with_weights-0
'layer-1
(layer_with_weights-1
(layer-2
)layer_with_weights-2
)layer-3
*layer-4
+layer_with_weights-3
+layer-5
,layer_with_weights-4
,layer-6
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_network
(
3	keras_api"
_tf_keras_layer
(
4	keras_api"
_tf_keras_layer
(
5	keras_api"
_tf_keras_layer
(
6	keras_api"
_tf_keras_layer
(
7	keras_api"
_tf_keras_layer
(
8	keras_api"
_tf_keras_layer
(
9	keras_api"
_tf_keras_layer
(
:	keras_api"
_tf_keras_layer
�
;layer-0
<layer_with_weights-0
<layer-1
=layer-2
>layer_with_weights-1
>layer-3
?layer_with_weights-2
?layer-4
@layer_with_weights-3
@layer-5
Alayer_with_weights-4
Alayer-6
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_network
(
H	keras_api"
_tf_keras_layer
(
I	keras_api"
_tf_keras_layer
(
J	keras_api"
_tf_keras_layer
(
K	keras_api"
_tf_keras_layer
(
L	keras_api"
_tf_keras_layer
(
M	keras_api"
_tf_keras_layer
(
N	keras_api"
_tf_keras_layer
(
O	keras_api"
_tf_keras_layer
(
P	keras_api"
_tf_keras_layer
(
Q	keras_api"
_tf_keras_layer
(
R	keras_api"
_tf_keras_layer
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
(
Y	keras_api"
_tf_keras_layer
�
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses"
_tf_keras_layer
(
`	keras_api"
_tf_keras_layer
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
�
m0
n1
o2
p3
q4
r5
s6
t7
u8
v9
w10
x11
y12
z13
{14
|15
}16
~17
18
�19"
trackable_list_wrapper
�
m0
n1
o2
p3
q4
r5
s6
t7
u8
v9
w10
x11
y12
z13
{14
|15
}16
~17
18
�19"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
#_default_save_signature
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
(__inference_model_2_layer_call_fn_239789
(__inference_model_2_layer_call_fn_240515
(__inference_model_2_layer_call_fn_240577
(__inference_model_2_layer_call_fn_240136�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
C__inference_model_2_layer_call_and_return_conditional_losses_240803
C__inference_model_2_layer_call_and_return_conditional_losses_241029
C__inference_model_2_layer_call_and_return_conditional_losses_240260
C__inference_model_2_layer_call_and_return_conditional_losses_240384�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�
capture_20
�
capture_23B�
!__inference__wrapped_model_238741input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
capture_20z�
capture_23
�
	�iter
�beta_1
�beta_2

�decay
�learning_ratemm�nm�om�pm�qm�rm�sm�tm�um�vm�wm�xm�ym�zm�{m�|m�}m�~m�m�	�m�mv�nv�ov�pv�qv�rv�sv�tv�uv�vv�wv�xv�yv�zv�{v�|v�}v�~v�v�	�v�"
	optimizer
 "
trackable_dict_wrapper
-
�serving_default"
signature_map
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

mkernel
nbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

okernel
pbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

qkernel
rbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

skernel
tbias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

ukernel
vbias"
_tf_keras_layer
f
m0
n1
o2
p3
q4
r5
s6
t7
u8
v9"
trackable_list_wrapper
f
m0
n1
o2
p3
q4
r5
s6
t7
u8
v9"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
&__inference_model_layer_call_fn_238866
&__inference_model_layer_call_fn_241056
&__inference_model_layer_call_fn_241083
&__inference_model_layer_call_fn_239034�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
A__inference_model_layer_call_and_return_conditional_losses_241123
A__inference_model_layer_call_and_return_conditional_losses_241163
A__inference_model_layer_call_and_return_conditional_losses_239065
A__inference_model_layer_call_and_return_conditional_losses_239096�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_tf_keras_input_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

wkernel
xbias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

ykernel
zbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

{kernel
|bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

}kernel
~bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
g
w0
x1
y2
z3
{4
|5
}6
~7
8
�9"
trackable_list_wrapper
g
w0
x1
y2
z3
{4
|5
}6
~7
8
�9"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
(__inference_model_1_layer_call_fn_239360
(__inference_model_1_layer_call_fn_241188
(__inference_model_1_layer_call_fn_241213
(__inference_model_1_layer_call_fn_239481�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
C__inference_model_1_layer_call_and_return_conditional_losses_241314
C__inference_model_1_layer_call_and_return_conditional_losses_241415
C__inference_model_1_layer_call_and_return_conditional_losses_239511
C__inference_model_1_layer_call_and_return_conditional_losses_239541�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_add_loss_layer_call_fn_241421�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_add_loss_layer_call_and_return_conditional_losses_241426�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_add_metric_layer_call_fn_241435�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_add_metric_layer_call_and_return_conditional_losses_241452�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_add_metric_1_layer_call_fn_241461�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_add_metric_1_layer_call_and_return_conditional_losses_241478�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_add_metric_2_layer_call_fn_241487�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_add_metric_2_layer_call_and_return_conditional_losses_241504�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
':%2conv2d/kernel
:2conv2d/bias
):'2conv2d_1/kernel
:2conv2d_1/bias
):' 2conv2d_2/kernel
: 2conv2d_2/bias
 :
��@2dense/kernel
:@2
dense/bias
": 
��@2dense_1/kernel
:@2dense_1/bias
": 
@��2dense_2/kernel
:��2dense_2/bias
1:/  2conv2d_transpose/kernel
#:! 2conv2d_transpose/bias
3:1 2conv2d_transpose_1/kernel
%:#2conv2d_transpose_1/bias
3:12conv2d_transpose_2/kernel
%:#2conv2d_transpose_2/bias
3:12conv2d_transpose_3/kernel
%:#2conv2d_transpose_3/bias
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
�
capture_20
�
capture_23B�
(__inference_model_2_layer_call_fn_239789input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
capture_20z�
capture_23
�
�
capture_20
�
capture_23B�
(__inference_model_2_layer_call_fn_240515inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
capture_20z�
capture_23
�
�
capture_20
�
capture_23B�
(__inference_model_2_layer_call_fn_240577inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
capture_20z�
capture_23
�
�
capture_20
�
capture_23B�
(__inference_model_2_layer_call_fn_240136input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
capture_20z�
capture_23
�
�
capture_20
�
capture_23B�
C__inference_model_2_layer_call_and_return_conditional_losses_240803inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
capture_20z�
capture_23
�
�
capture_20
�
capture_23B�
C__inference_model_2_layer_call_and_return_conditional_losses_241029inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
capture_20z�
capture_23
�
�
capture_20
�
capture_23B�
C__inference_model_2_layer_call_and_return_conditional_losses_240260input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
capture_20z�
capture_23
�
�
capture_20
�
capture_23B�
C__inference_model_2_layer_call_and_return_conditional_losses_240384input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
capture_20z�
capture_23
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�
�
capture_20
�
capture_23B�
$__inference_signature_wrapper_240453input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
capture_20z�
capture_23
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv2d_layer_call_fn_241513�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_conv2d_layer_call_and_return_conditional_losses_241524�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
o0
p1"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_1_layer_call_fn_241533�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_241544�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_2_layer_call_fn_241553�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_241564�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_flatten_layer_call_fn_241569�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_flatten_layer_call_and_return_conditional_losses_241575�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_layer_call_fn_241584�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_layer_call_and_return_conditional_losses_241594�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
u0
v1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_1_layer_call_fn_241603�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_1_layer_call_and_return_conditional_losses_241613�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
Q
0
'1
(2
)3
*4
+5
,6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_model_layer_call_fn_238866input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_model_layer_call_fn_241056inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_model_layer_call_fn_241083inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_model_layer_call_fn_239034input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_model_layer_call_and_return_conditional_losses_241123inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_model_layer_call_and_return_conditional_losses_241163inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_model_layer_call_and_return_conditional_losses_239065input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_model_layer_call_and_return_conditional_losses_239096input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_2_layer_call_fn_241622�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_2_layer_call_and_return_conditional_losses_241633�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_reshape_layer_call_fn_241638�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_reshape_layer_call_and_return_conditional_losses_241652�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_conv2d_transpose_layer_call_fn_241661�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_241695�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_conv2d_transpose_1_layer_call_fn_241704�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_241738�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_conv2d_transpose_2_layer_call_fn_241747�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_241781�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
/
0
�1"
trackable_list_wrapper
/
0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_conv2d_transpose_3_layer_call_fn_241790�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_241824�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
Q
;0
<1
=2
>3
?4
@5
A6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_model_1_layer_call_fn_239360input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_model_1_layer_call_fn_241188inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_model_1_layer_call_fn_241213inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_model_1_layer_call_fn_239481input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_1_layer_call_and_return_conditional_losses_241314inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_1_layer_call_and_return_conditional_losses_241415inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_1_layer_call_and_return_conditional_losses_239511input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_1_layer_call_and_return_conditional_losses_239541input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_add_loss_layer_call_fn_241421inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_add_loss_layer_call_and_return_conditional_losses_241426inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
:
�reconstruction_loss"
trackable_dict_wrapper
�B�
+__inference_add_metric_layer_call_fn_241435inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_add_metric_layer_call_and_return_conditional_losses_241452inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
�kl_loss"
trackable_dict_wrapper
�B�
-__inference_add_metric_1_layer_call_fn_241461inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_add_metric_1_layer_call_and_return_conditional_losses_241478inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�	elbo_loss"
trackable_dict_wrapper
�B�
-__inference_add_metric_2_layer_call_fn_241487inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_add_metric_2_layer_call_and_return_conditional_losses_241504inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_conv2d_layer_call_fn_241513inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_conv2d_layer_call_and_return_conditional_losses_241524inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv2d_1_layer_call_fn_241533inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_241544inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv2d_2_layer_call_fn_241553inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_241564inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_flatten_layer_call_fn_241569inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_flatten_layer_call_and_return_conditional_losses_241575inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_dense_layer_call_fn_241584inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_layer_call_and_return_conditional_losses_241594inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_1_layer_call_fn_241603inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_1_layer_call_and_return_conditional_losses_241613inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_2_layer_call_fn_241622inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_2_layer_call_and_return_conditional_losses_241633inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_reshape_layer_call_fn_241638inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_reshape_layer_call_and_return_conditional_losses_241652inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_conv2d_transpose_layer_call_fn_241661inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_241695inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_conv2d_transpose_1_layer_call_fn_241704inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_241738inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_conv2d_transpose_2_layer_call_fn_241747inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_241781inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_conv2d_transpose_3_layer_call_fn_241790inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_241824inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2add_metric/total
:  (2add_metric/count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2add_metric_1/total
:  (2add_metric_1/count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2add_metric_2/total
:  (2add_metric_2/count
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
.:,2Adam/conv2d_1/kernel/m
 :2Adam/conv2d_1/bias/m
.:, 2Adam/conv2d_2/kernel/m
 : 2Adam/conv2d_2/bias/m
%:#
��@2Adam/dense/kernel/m
:@2Adam/dense/bias/m
':%
��@2Adam/dense_1/kernel/m
:@2Adam/dense_1/bias/m
':%
@��2Adam/dense_2/kernel/m
!:��2Adam/dense_2/bias/m
6:4  2Adam/conv2d_transpose/kernel/m
(:& 2Adam/conv2d_transpose/bias/m
8:6 2 Adam/conv2d_transpose_1/kernel/m
*:(2Adam/conv2d_transpose_1/bias/m
8:62 Adam/conv2d_transpose_2/kernel/m
*:(2Adam/conv2d_transpose_2/bias/m
8:62 Adam/conv2d_transpose_3/kernel/m
*:(2Adam/conv2d_transpose_3/bias/m
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
.:,2Adam/conv2d_1/kernel/v
 :2Adam/conv2d_1/bias/v
.:, 2Adam/conv2d_2/kernel/v
 : 2Adam/conv2d_2/bias/v
%:#
��@2Adam/dense/kernel/v
:@2Adam/dense/bias/v
':%
��@2Adam/dense_1/kernel/v
:@2Adam/dense_1/bias/v
':%
@��2Adam/dense_2/kernel/v
!:��2Adam/dense_2/bias/v
6:4  2Adam/conv2d_transpose/kernel/v
(:& 2Adam/conv2d_transpose/bias/v
8:6 2 Adam/conv2d_transpose_1/kernel/v
*:(2Adam/conv2d_transpose_1/bias/v
8:62 Adam/conv2d_transpose_2/kernel/v
*:(2Adam/conv2d_transpose_2/bias/v
8:62 Adam/conv2d_transpose_3/kernel/v
*:(2Adam/conv2d_transpose_3/bias/v�
!__inference__wrapped_model_238741�%mnopqruvstwxyz{|}~���������:�7
0�-
+�(
input_1�����������
� ";�8
6
model_1+�(
model_1������������
D__inference_add_loss_layer_call_and_return_conditional_losses_241426D�
�
�
inputs 
� ""�

�
0 
�
�	
1/0 V
)__inference_add_loss_layer_call_fn_241421)�
�
�
inputs 
� "� �
H__inference_add_metric_1_layer_call_and_return_conditional_losses_241478<���
�
�
inputs 
� "�

�
0 
� `
-__inference_add_metric_1_layer_call_fn_241461/���
�
�
inputs 
� "� �
H__inference_add_metric_2_layer_call_and_return_conditional_losses_241504<���
�
�
inputs 
� "�

�
0 
� `
-__inference_add_metric_2_layer_call_fn_241487/���
�
�
inputs 
� "� �
F__inference_add_metric_layer_call_and_return_conditional_losses_241452<���
�
�
inputs 
� "�

�
0 
� ^
+__inference_add_metric_layer_call_fn_241435/���
�
�
inputs 
� "� �
D__inference_conv2d_1_layer_call_and_return_conditional_losses_241544lop7�4
-�*
(�%
inputs���������pp
� "-�*
#� 
0���������88
� �
)__inference_conv2d_1_layer_call_fn_241533_op7�4
-�*
(�%
inputs���������pp
� " ����������88�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_241564lqr7�4
-�*
(�%
inputs���������88
� "-�*
#� 
0��������� 
� �
)__inference_conv2d_2_layer_call_fn_241553_qr7�4
-�*
(�%
inputs���������88
� " ���������� �
B__inference_conv2d_layer_call_and_return_conditional_losses_241524nmn9�6
/�,
*�'
inputs�����������
� "-�*
#� 
0���������pp
� �
'__inference_conv2d_layer_call_fn_241513amn9�6
/�,
*�'
inputs�����������
� " ����������pp�
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_241738�{|I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+���������������������������
� �
3__inference_conv2d_transpose_1_layer_call_fn_241704�{|I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+����������������������������
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_241781�}~I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
3__inference_conv2d_transpose_2_layer_call_fn_241747�}~I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_241824��I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
3__inference_conv2d_transpose_3_layer_call_fn_241790��I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_241695�yzI�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+��������������������������� 
� �
1__inference_conv2d_transpose_layer_call_fn_241661�yzI�F
?�<
:�7
inputs+��������������������������� 
� "2�/+��������������������������� �
C__inference_dense_1_layer_call_and_return_conditional_losses_241613^uv1�.
'�$
"�
inputs�����������
� "%�"
�
0���������@
� }
(__inference_dense_1_layer_call_fn_241603Quv1�.
'�$
"�
inputs�����������
� "����������@�
C__inference_dense_2_layer_call_and_return_conditional_losses_241633^wx/�,
%�"
 �
inputs���������@
� "'�$
�
0�����������
� }
(__inference_dense_2_layer_call_fn_241622Qwx/�,
%�"
 �
inputs���������@
� "�������������
A__inference_dense_layer_call_and_return_conditional_losses_241594^st1�.
'�$
"�
inputs�����������
� "%�"
�
0���������@
� {
&__inference_dense_layer_call_fn_241584Qst1�.
'�$
"�
inputs�����������
� "����������@�
C__inference_flatten_layer_call_and_return_conditional_losses_241575b7�4
-�*
(�%
inputs��������� 
� "'�$
�
0�����������
� �
(__inference_flatten_layer_call_fn_241569U7�4
-�*
(�%
inputs��������� 
� "�������������
C__inference_model_1_layer_call_and_return_conditional_losses_239511xwxyz{|}~�8�5
.�+
!�
input_2���������@
p 

 
� "/�,
%�"
0�����������
� �
C__inference_model_1_layer_call_and_return_conditional_losses_239541xwxyz{|}~�8�5
.�+
!�
input_2���������@
p

 
� "/�,
%�"
0�����������
� �
C__inference_model_1_layer_call_and_return_conditional_losses_241314wwxyz{|}~�7�4
-�*
 �
inputs���������@
p 

 
� "/�,
%�"
0�����������
� �
C__inference_model_1_layer_call_and_return_conditional_losses_241415wwxyz{|}~�7�4
-�*
 �
inputs���������@
p

 
� "/�,
%�"
0�����������
� �
(__inference_model_1_layer_call_fn_239360kwxyz{|}~�8�5
.�+
!�
input_2���������@
p 

 
� ""�������������
(__inference_model_1_layer_call_fn_239481kwxyz{|}~�8�5
.�+
!�
input_2���������@
p

 
� ""�������������
(__inference_model_1_layer_call_fn_241188jwxyz{|}~�7�4
-�*
 �
inputs���������@
p 

 
� ""�������������
(__inference_model_1_layer_call_fn_241213jwxyz{|}~�7�4
-�*
 �
inputs���������@
p

 
� ""�������������
C__inference_model_2_layer_call_and_return_conditional_losses_240260�%mnopqruvstwxyz{|}~���������B�?
8�5
+�(
input_1�����������
p 

 
� "=�:
%�"
0�����������
�
�	
1/0 �
C__inference_model_2_layer_call_and_return_conditional_losses_240384�%mnopqruvstwxyz{|}~���������B�?
8�5
+�(
input_1�����������
p

 
� "=�:
%�"
0�����������
�
�	
1/0 �
C__inference_model_2_layer_call_and_return_conditional_losses_240803�%mnopqruvstwxyz{|}~���������A�>
7�4
*�'
inputs�����������
p 

 
� "=�:
%�"
0�����������
�
�	
1/0 �
C__inference_model_2_layer_call_and_return_conditional_losses_241029�%mnopqruvstwxyz{|}~���������A�>
7�4
*�'
inputs�����������
p

 
� "=�:
%�"
0�����������
�
�	
1/0 �
(__inference_model_2_layer_call_fn_239789�%mnopqruvstwxyz{|}~���������B�?
8�5
+�(
input_1�����������
p 

 
� ""�������������
(__inference_model_2_layer_call_fn_240136�%mnopqruvstwxyz{|}~���������B�?
8�5
+�(
input_1�����������
p

 
� ""�������������
(__inference_model_2_layer_call_fn_240515�%mnopqruvstwxyz{|}~���������A�>
7�4
*�'
inputs�����������
p 

 
� ""�������������
(__inference_model_2_layer_call_fn_240577�%mnopqruvstwxyz{|}~���������A�>
7�4
*�'
inputs�����������
p

 
� ""�������������
A__inference_model_layer_call_and_return_conditional_losses_239065�
mnopqruvstB�?
8�5
+�(
input_1�����������
p 

 
� "K�H
A�>
�
0/0���������@
�
0/1���������@
� �
A__inference_model_layer_call_and_return_conditional_losses_239096�
mnopqruvstB�?
8�5
+�(
input_1�����������
p

 
� "K�H
A�>
�
0/0���������@
�
0/1���������@
� �
A__inference_model_layer_call_and_return_conditional_losses_241123�
mnopqruvstA�>
7�4
*�'
inputs�����������
p 

 
� "K�H
A�>
�
0/0���������@
�
0/1���������@
� �
A__inference_model_layer_call_and_return_conditional_losses_241163�
mnopqruvstA�>
7�4
*�'
inputs�����������
p

 
� "K�H
A�>
�
0/0���������@
�
0/1���������@
� �
&__inference_model_layer_call_fn_238866�
mnopqruvstB�?
8�5
+�(
input_1�����������
p 

 
� "=�:
�
0���������@
�
1���������@�
&__inference_model_layer_call_fn_239034�
mnopqruvstB�?
8�5
+�(
input_1�����������
p

 
� "=�:
�
0���������@
�
1���������@�
&__inference_model_layer_call_fn_241056�
mnopqruvstA�>
7�4
*�'
inputs�����������
p 

 
� "=�:
�
0���������@
�
1���������@�
&__inference_model_layer_call_fn_241083�
mnopqruvstA�>
7�4
*�'
inputs�����������
p

 
� "=�:
�
0���������@
�
1���������@�
C__inference_reshape_layer_call_and_return_conditional_losses_241652b1�.
'�$
"�
inputs�����������
� "-�*
#� 
0��������� 
� �
(__inference_reshape_layer_call_fn_241638U1�.
'�$
"�
inputs�����������
� " ���������� �
$__inference_signature_wrapper_240453�%mnopqruvstwxyz{|}~���������E�B
� 
;�8
6
input_1+�(
input_1�����������";�8
6
model_1+�(
model_1�����������
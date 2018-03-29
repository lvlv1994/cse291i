tensorflow = /Users/chunyilyu/anaconda2/lib/python2.7/site-packages/tensorflow/include
tensorflow_lib = /Users/chunyilyu/anaconda2/lib/python2.7/site-packages/tensorflow
all: tf_emddistance_so.so 
.PHONY : all

tf_emddistance_so.so:  tf_emddistance.cpp
	g++ -std=c++11 tf_emddistance.cpp  -o tf_emddistance_so.so -shared -fPIC -I $(tensorflow) -I $(tensorflow)/external/nsync/public -ltensorflow_framework -L $(tensorflow_lib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0

TEMPLATE = app
CONFIG += console c++11
CONFIG += c++14 c++17
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        classifier.cpp \
        facenet.cpp \
        main.cpp \
        model.cpp

TORCHDIR = c:/SDKs/libtorch_171
TORCHDIR_DEBUG = c:/SDKs/libtorch_debug_171

# libtorch
INCLUDEPATH += $${TORCHDIR}/include
INCLUDEPATH += $${TORCHDIR}/include/torch/csrc/api/include
win32:CONFIG(release, debug|release): {
    LIBS += -L"$${TORCHDIR}/lib"
}
else:win32:CONFIG(debug, debug|release): {
    LIBS += -L"$${TORCHDIR_DEBUG}/lib"
}
LIBS += \
    asmjit.lib \
    c10.lib \
    c10_cuda.lib \
    caffe2_detectron_ops_gpu.lib \
    caffe2_module_test_dynamic.lib \
    caffe2_nvrtc.lib \
    clog.lib \
    cpuinfo.lib \
    dnnl.lib \
    fbgemm.lib \
    #libprotobuf-lite.lib \
    #libprotobuf.lib \
    #libprotoc.lib \
    #mkldnn.lib \
    torch.lib \
    torch_cpu.lib \
    torch_cuda.lib
# to force linker use cuda version of the library
LIBS += -INCLUDE:?warp_size@cuda@at@@YAHXZ

HEADERS += \
    classifier.h \
    facenet.h \
    model.h

#opencv
LIBS += -LC:\SDKs\opencv_build\install\x64\vc16\lib -lopencv_core450d -lopencv_imgproc450d -lopencv_highgui450d -lopencv_imgcodecs450d -lopencv_videoio450d -lopencv_video450d -lopencv_calib3d450d -lopencv_photo450d -lopencv_features2d450d
LIBS += -LC:\SDKs\opencv_build\install\x64\vc16\lib -lopencv_core450d -lopencv_imgproc450d -lopencv_highgui450d -lopencv_imgcodecs450d -lopencv_videoio450d -lopencv_video450d -lopencv_calib3d450d -lopencv_photo450d -lopencv_features2d450d

INCLUDEPATH += C:\SDKs\opencv_build\install\include
DEPENDPATH += C:\SDKs\opencv_build\install\include

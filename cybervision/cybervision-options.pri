CYBERVISION_BUILD_OPTIONS =
#CYBERVISION_BUILD_OPTIONS = noOpenCL
#CYBERVISION_BUILD_OPTIONS = noOpenCL noSSE

CYBERVISION_SSE = false
CYBERVISION_OPENCL = false

contains(CYBERVISION_BUILD_OPTIONS,"noSSE"){
    #message("SSE disabled in qmake")
}else{
    #message("SSE enabled in qmake")
    CYBERVISION_SSE = true
}

contains(CYBERVISION_BUILD_OPTIONS,"noOpenCL"){
    #message("OpenCL disabled in qmake")
}else{
    #message("OpenCL enabled in qmake")
    CYBERVISION_OPENCL = true
}

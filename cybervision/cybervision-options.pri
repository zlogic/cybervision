CYBERVISION_BUILD_OPTIONS =
#CYBERVISION_BUILD_OPTIONS = noOpenCL
#CYBERVISION_BUILD_OPTIONS = noOpenCL noSSE
#CYBERVISION_BUILD_OPTIONS = Demo
#CYBERVISION_BUILD_OPTIONS = Demo noOpenCL
#CYBERVISION_BUILD_OPTIONS = Demo noOpenCL noSSE

CYBERVISION_SSE = false
CYBERVISION_OPENCL = false
CYBERVISION_DEMO = false

CYBERVISION_SUFFIX= $$join(CYBERVISION_BUILD_OPTIONS,"_","_")
#message("CYBERVISION_SUFFIX= $${CYBERVISION_SUFFIX}")

contains(CYBERVISION_BUILD_OPTIONS,"Demo"){
	#message("Demo mode enabled in qmake")
	CYBERVISION_DEMO = true
}else{
	#message("Demo mode disabled in qmake")
}

contains(CYBERVISION_BUILD_OPTIONS,"noOpenCL"){
	#message("OpenCL disabled in qmake")
}else{
	#message("OpenCL enabled in qmake")
	CYBERVISION_OPENCL = true
}

contains(CYBERVISION_BUILD_OPTIONS,"noSSE"){
    #message("SSE disabled in qmake")
}else{
    #message("SSE enabled in qmake")
	CYBERVISION_SSE = true
}

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11
CONFIG += static
# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    mainwindow.h

FORMS += \
    mainwindow.ui

# OpenCV
#INCLUDEPATH += /usr/local/include/opencv4
#LIBS += $(shell pkg-config opencv --libs)\
#-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_dnn

#LIBS += -LC:/workspace/build-carclass_client-Desktop_Qt_6_0_2_MSVC2019_64bit-Debug/debug/opencv_core451.dll
#LIBS += -LC:/workspace/build-carclass_client-Desktop_Qt_6_0_2_MSVC2019_64bit-Debug/debug/opencv_highgui451.dll
#LIBS += -LC:/workspace/build-carclass_client-Desktop_Qt_6_0_2_MSVC2019_64bit-Debug/debug/opencv_imgproc451.dll
#LIBS += -LC:/workspace/build-carclass_client-Desktop_Qt_6_0_2_MSVC2019_64bit-Debug/debug/opencv_imgcodecs451.dll
#LIBS += -LC:/workspace/build-carclass_client-Desktop_Qt_6_0_2_MSVC2019_64bit-Debug/debug/opencv_videoio451.dll
#LIBS += -LC:/workspace/build-carclass_client-Desktop_Qt_6_0_2_MSVC2019_64bit-Debug/debug/opencv_dnn451.dll

#LIBS += -LC:/opencv/opencv-4.5.1/build/install/x64/vc16/bin/opencv_core451.dll
#LIBS += -LC:/opencv/opencv-4.5.1/build/install/x64/vc16/bin/opencv_highgui451.dll
#LIBS += -LC:/opencv/opencv-4.5.1/build/install/x64/vc16/bin/opencv_imgproc451.dll
#LIBS += -LC:/opencv/opencv-4.5.1/build/install/x64/vc16/bin/opencv_imgcodecs451.dll
#LIBS += -LC:/opencv/opencv-4.5.1/build/install/x64/vc16/bin/opencv_videoio451.dll
#LIBS += -LC:/opencv/opencv-4.5.1/build/install/x64/vc16/bin/opencv_dnn451.dll

INCLUDEPATH += C:/opencv/opencv-4.5.1/build/install/include

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16/lib/ -lopencv_core451
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16/lib/ -lopencv_core451d
else:unix: LIBS += -L$$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16/lib/ -lopencv_core451

INCLUDEPATH += $$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16
DEPENDPATH += $$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16/lib/ -lopencv_dnn451
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16/lib/ -lopencv_dnn451d
else:unix: LIBS += -L$$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16/lib/ -lopencv_dnn451

INCLUDEPATH += $$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16
DEPENDPATH += $$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16/lib/ -lopencv_highgui451
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16/lib/ -lopencv_highgui451d
else:unix: LIBS += -L$$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16/lib/ -lopencv_highgui451

INCLUDEPATH += $$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16
DEPENDPATH += $$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16/lib/ -lopencv_imgproc451
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16/lib/ -lopencv_imgproc451d
else:unix: LIBS += -L$$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16/lib/ -lopencv_imgproc451

INCLUDEPATH += $$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16
DEPENDPATH += $$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16/lib/ -lopencv_imgcodecs451
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16/lib/ -lopencv_imgcodecs451d
else:unix: LIBS += -L$$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16/lib/ -lopencv_imgcodecs451

INCLUDEPATH += $$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16
DEPENDPATH += $$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16/lib/ -lopencv_videoio451
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16/lib/ -lopencv_videoio451d
else:unix: LIBS += -L$$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16/lib/ -lopencv_videoio451

INCLUDEPATH += $$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16
DEPENDPATH += $$PWD/../../opencv/opencv-4.5.1/build/install/x64/vc16

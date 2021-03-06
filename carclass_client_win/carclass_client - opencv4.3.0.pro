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


INCLUDEPATH += C:/opencv/opencv-4.3.0/build/install/include

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16/lib/ -lopencv_core430
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16/lib/ -lopencv_core430d
else:unix: LIBS += -L$$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16/lib/ -lopencv_core430

INCLUDEPATH += $$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16
DEPENDPATH += $$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16/lib/ -lopencv_dnn430
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16/lib/ -lopencv_dnn430d
else:unix: LIBS += -L$$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16/lib/ -lopencv_dnn430

INCLUDEPATH += $$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16
DEPENDPATH += $$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16/lib/ -lopencv_highgui430
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16/lib/ -lopencv_highgui430d
else:unix: LIBS += -L$$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16/lib/ -lopencv_highgui430

INCLUDEPATH += $$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16
DEPENDPATH += $$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16/lib/ -lopencv_imgproc430
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16/lib/ -lopencv_imgproc430d
else:unix: LIBS += -L$$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16/lib/ -lopencv_imgproc430

INCLUDEPATH += $$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16
DEPENDPATH += $$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16/lib/ -lopencv_imgcodecs430
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16/lib/ -lopencv_imgcodecs430d
else:unix: LIBS += -L$$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16/lib/ -lopencv_imgcodecs430

INCLUDEPATH += $$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16
DEPENDPATH += $$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16/lib/ -lopencv_videoio430
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16/lib/ -lopencv_videoio430d
else:unix: LIBS += -L$$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16/lib/ -lopencv_videoio430

INCLUDEPATH += $$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16
DEPENDPATH += $$PWD/../../opencv/opencv-4.3.0/build/install/x64/vc16

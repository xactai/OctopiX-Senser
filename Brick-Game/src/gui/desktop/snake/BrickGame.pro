QT       += core gui printsupport

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

SOURCES += \
    ./main.cc \
    ./View.cc \
    ../../brick_game/tetris/TetrisModel.cc \
    ../../brick_game/tetris/Tetromino.cc


HEADERS += \
    ./View.h \
    ../../brick_game/tetris/TetrisModel.h \
    ../../brick_game/tetris/Tetromino.h \
    ../../controller/TetrisController.h

FORMS += \
        View.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

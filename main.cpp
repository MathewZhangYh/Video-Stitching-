#include "video_stitch.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    video_stitch w;
    w.show();
    return a.exec();
}

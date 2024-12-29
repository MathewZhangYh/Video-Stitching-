#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_video_stitch.h"

#include <QApplication>
#include <QLabel>
#include <QTimer>
#include <QDateTime>
#include <QThread>
#include <QDebug>

#include <QFileDialog>
#include <opencv2/opencv.hpp>
#include "image_stitch.h"
class video_stitch : public QMainWindow
{
    Q_OBJECT

public:
    video_stitch(QWidget *parent = nullptr);
    ~video_stitch();
	int videowork(std::string Name_1, std::string Name_2, std::string Name_3);
private slots:
	void onClicked_videostitch();

private:
    Ui::video_stitchClass ui;
};

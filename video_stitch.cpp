#include "video_stitch.h"

video_stitch::video_stitch(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

	connect(ui.pushButton, SIGNAL(clicked()), this, SLOT(onClicked_videostitch()));
}

video_stitch::~video_stitch()
{}

int video_stitch::videowork(std::string Name_1, std::string Name_2, std::string Name_3)
{
	//录像
	cv::VideoWriter writerResult;
	int codec = writerResult.fourcc('X', 'V', 'I', 'D');
	double fps = 29.0;
	QDateTime dateTime = QDateTime::currentDateTime();
	QString str = dateTime.toString("yyyy-MM-dd_hhmmss");
	QString Name = "./" + str + ".avi";
	std::string recordName = Name.toStdString();
	std::cout << recordName << std::endl;
	bool W = false;

	//输入视频流
	cv::VideoCapture cap1(Name_1);
	cv::VideoCapture cap2(Name_2);
	cv::VideoCapture cap3(Name_3);

	if (!cap1.isOpened() || !cap2.isOpened() || !cap3.isOpened())
	{
		std::cout << "could not open cameras" << std::endl;
		return -1;
	}
	static uint64_t out = 0;
	static uint64_t last = 0;
	static uint64_t oncetime = 0;
	cv::Mat image1, image2, image3, result_stitch;
	bool first = true;
	ImageStitch* imgstitch = new ImageStitch;
	while (true)
	{
		if (cap1.read(image1) && cap2.read(image2) && cap3.read(image3))
		{
			//控制帧率
			out = QDateTime::currentDateTime().toMSecsSinceEpoch();
			oncetime = out - last;
			last = out;
			if (oncetime < 41)
			{
				QThread::msleep(41 - oncetime);
				last = QDateTime::currentDateTime().toMSecsSinceEpoch();
			}
			//拼接
			if (first)
			{
				imgstitch->iniStitch(image1, image2, image3);
				first = false;
			}
			result_stitch = imgstitch->imageStitch(image1, image2, image3);

			if (!W)
			{
				writerResult.open(recordName, codec, fps, result_stitch.size(), true);
				W = true;
			}
			writerResult.write(result_stitch);//写
			//显示
			cv::cvtColor(image1, image1, cv::COLOR_BGR2RGB);
			cv::cvtColor(image2, image2, cv::COLOR_BGR2RGB);
			cv::cvtColor(image3, image3, cv::COLOR_BGR2RGB);
			cv::cvtColor(result_stitch, result_stitch, cv::COLOR_BGR2RGB);
			QImage Img = QImage((const uchar*)(image1.data), image1.cols, image1.rows, image1.cols * image1.channels(), QImage::Format_RGB888);
			QImage Img2 = QImage((const uchar*)(image2.data), image2.cols, image2.rows, image2.cols * image2.channels(), QImage::Format_RGB888);
			QImage Img3 = QImage((const uchar*)(image3.data), image3.cols, image3.rows, image3.cols * image3.channels(), QImage::Format_RGB888);
			QImage Img4 = QImage((const uchar*)(result_stitch.data), result_stitch.cols, result_stitch.rows, result_stitch.cols * result_stitch.channels(), QImage::Format_RGB888);
			ui.label->setPixmap(QPixmap::fromImage(Img).scaled(ui.label->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
			ui.label_2->setPixmap(QPixmap::fromImage(Img2).scaled(ui.label_2->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
			ui.label_3->setPixmap(QPixmap::fromImage(Img3).scaled(ui.label_3->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
			ui.label_4->setPixmap(QPixmap::fromImage(Img4).scaled(ui.label_4->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));

			cv::waitKey(1);

		}
		else
		{
			ui.label->clear();
			ui.label_2->clear();
			ui.label_3->clear();
			ui.label_4->clear();
			writerResult.release();
			std::cout << "处理完毕" << std::endl;
			break;
		}
	}
	return 0;
}

void video_stitch::onClicked_videostitch()
{
	QStringList filePath_list = QFileDialog::getOpenFileNames(this, "Open", "", "All File(*)");
	if (filePath_list.size() < 3)
	{
		std::cout << "无三路视频" << std::endl;
		return;
	}
	QStringList videoExtensions = { "mp4", "avi", "mov" };

	for (const QString &filePath : filePath_list)
	{
		QFileInfo fileInfo(filePath);
		QString extension = fileInfo.suffix().toLower();

		if (videoExtensions.contains(extension))
		{
			qDebug() << "Selected video file: " << filePath;
		}
		else
		{
			qDebug() << "Not a valid video file: " << filePath;
			return;
		}
	}
	std::string sfileName_1 = filePath_list[0].toStdString();
	std::string sfileName_2 = filePath_list[1].toStdString();
	std::string sfileName_3 = filePath_list[2].toStdString();

	videowork(sfileName_1, sfileName_2, sfileName_3);


}


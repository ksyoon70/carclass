#include "mainwindow.h"
#include "ui_mainwindow.h"

#define  Program_Name   "차량(대소)분류"
#define  Program_Version  "1.0.0"
#define  Program_Date   "2021/03/12"

#define REG_SUCCESS (80.0)

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QString title = QString("%1 v%2 (date: %3)").arg(Program_Name).arg(Program_Version).arg(Program_Date);

    this->setWindowTitle(title);

    setFixedSize(this->geometry().width(),this->geometry().height());

    try {
        net = readNet(".\\frozen_models\\frozen_graph.pb");
        if(net.empty())
        {
            return;
        }
    }  catch (...) {
         qDebug() << "reading frozen_graph.pb.";
    }



    m_startPath = QDir::rootPath(); //QDir::homePath(); //QDir::rootPath();
    treeModel = new QFileSystemModel;
    //treeModel->setFilter(QDir::NoDotAndDotDot | QDir::AllDirs);
    treeModel->setRootPath(m_startPath);
    ui->treeView->setModel(treeModel);
    for (int i = 1; i < treeModel->columnCount(); ++i)
        ui->treeView->hideColumn(i);

    QModelIndex index = treeModel->index(m_startPath, 0);
    ui->treeView->setRootIndex(index);


    QString sPath = m_startPath;
    model = new QFileSystemModel;
    model->setRootPath(sPath);
    ui->listView->setModel(model);
    //ui->listView->setRootIndex(model->index(sPath));
    ui->listView->setEditTriggers(QAbstractItemView::AnyKeyPressed |QAbstractItemView::DoubleClicked );
    QItemSelectionModel *selectionModel = ui->listView->selectionModel();
    if(!connect( selectionModel, SIGNAL( currentRowChanged(QModelIndex,QModelIndex)), this, SLOT( rowChangedSlot(QModelIndex,QModelIndex) ) ))
    {
        qDebug("Something wrong :(");
    }

    ui->listView->show();

    //label 크기 고정
    int w = ui->lbImage->width();
    int h = ui->lbImage->height();
    ui->lbImage->setFixedWidth(w);
    ui->lbImage->setFixedHeight(h);

    sPath = treeModel->fileInfo(index).absoluteFilePath();
    ui->listView->setRootIndex(model->index(sPath));
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_treeView_clicked(const QModelIndex &index)
{
    QString sPath = treeModel->fileInfo(index).absoluteFilePath();
    ui->listView->setRootIndex(model->index(sPath));
}

void MainWindow::on_listView_clicked(const QModelIndex &index)
{
    if(model->fileInfo(index).isFile())
    {
        //선택한것이 파일이면.
        QString ext = model->fileInfo(index).completeSuffix();
        ext = ext.toLower();
        QString filename = model->fileInfo(index).absoluteFilePath();
        if(ext == "jpg")
        {
            cv::Mat imat;
            imat = cv::imread(filename.toLocal8Bit().toStdString(), cv::IMREAD_COLOR);  //한글폴더가 포함될 경우 오류 수정
            QImage image = QImage((unsigned char*) imat.data, imat.cols, imat.rows, QImage::Format_RGB888);
            int w = ui->lbImage->width();
            int h = ui->lbImage->height();
            ui->lbImage->setPixmap( QPixmap::fromImage(image.rgbSwapped()).scaled(w,h));
            ui->lbImage->setScaledContents(false);
            ui->lbImage->show();

            if(net.empty())
            {
                QString err;
                err = "dnn 이 초기화 되지 않았습니다";
                ui->lineEdit->setText(err);
                return;
            }


            try{

                cv::Mat dst;
                cv::resize(imat,dst, cv::Size(224, 224), 0,0, cv::INTER_LINEAR);

                Mat inputBlob = blobFromImage(dst,1/255.f, Size(224,224),Scalar(),true,false,CV_32F);
                net.setInput(inputBlob);
                Mat prob = net.forward();
                double maxVal;
                Point maxLoc;
                minMaxLoc(prob,NULL,&maxVal,NULL,&maxLoc);
                int digit = maxLoc.x;
                QString ret;
                if( maxVal*100 >=REG_SUCCESS)
                {
                    ret = QString("%1 --> 확률: %2 %").arg((digit == 0) ? "소형차량" :"대형차량" ).arg(QString::number(maxVal*100,'f',2));
                }
                else
                {
                    ret = QString("%1 --> 확률: %2 %").arg("인식실패").arg(QString::number(maxVal*100,'f',2));
                }
                ui->lineEdit->setText(ret);

            }catch(...)
            {
                QString err;
                err = QString("인식오류 %1 ").arg(filename);
                ui->lineEdit->setText(err);
                return;
            }



        }
    }
}

void MainWindow::on_listView_doubleClicked(const QModelIndex &index)
{
    if(model->fileInfo(index).isDir())
    {
        ui->listView->setRootIndex(index);
        QString path = model->fileInfo(index).absolutePath();
        //ui->treeView->setRootIndex(treeModel->index(path));
        ui->treeView->setExpanded(treeModel->index(path),true);
    }
}
void MainWindow::rowChangedSlot ( const QModelIndex & current, const QModelIndex & previous )
{
        on_listView_clicked(current);
}

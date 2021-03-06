#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
//#include <QDirModel>
#include <QListView>
#include <QFileSystemModel>


#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_treeView_clicked(const QModelIndex &index);

    void on_listView_clicked(const QModelIndex &index);

    void on_listView_doubleClicked(const QModelIndex &index);

private:
    Ui::MainWindow *ui;
    QListView *list;
    QFileSystemModel *model;
    QFileSystemModel * treeModel;
    QString m_startPath;
    Net net;
 public:
    bool ListViewKeyEvent(QObject *obj, QEvent *event);
 private slots:
    void rowChangedSlot ( const QModelIndex & current, const QModelIndex & previous );
};
#endif // MAINWINDOW_H

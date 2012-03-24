#include "aboutwindow.h"
#include "ui_aboutwindow.h"
#include <QDesktopServices>
#include <QUrl>

AboutWindow::AboutWindow(QWidget *parent) : QDialog(parent),ui(new Ui::AboutWindow){
	ui->setupUi(this);
}

AboutWindow::~AboutWindow(){
	delete ui;
}

void AboutWindow::on_closeButton_clicked(){
	close();
}

void AboutWindow::on_copyrightLabel_linkActivated(const QString &link){
	QDesktopServices::openUrl(QUrl(link));
}

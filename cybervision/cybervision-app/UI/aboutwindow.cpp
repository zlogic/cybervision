#include "aboutwindow.h"
#include "ui_aboutwindow.h"
#include <QDesktopServices>
#include <QUrl>

AboutWindow::AboutWindow(QWidget *parent) : QDialog(parent),ui(new Ui::AboutWindow){
	ui->setupUi(this);

	connect(ui->closeButton,&QPushButton::clicked,this,&AboutWindow::close);
	connect(ui->copyrightLabel,&QLabel::linkActivated,this,&AboutWindow::openLink);
}

AboutWindow::~AboutWindow(){
	delete ui;
}

void AboutWindow::openLink(const QString &link){
	QDesktopServices::openUrl(QUrl(link));
}

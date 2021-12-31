#include "aboutwindow.h"
#include "ui_aboutwindow.h"
#include <QDesktopServices>
#include <QUrl>

AboutWindow::AboutWindow(QWidget *parent) : QDialog(parent),ui(new Ui::AboutWindow){
	ui->setupUi(this);

	connect(ui->closeButton, SIGNAL(clicked()), this, SLOT(close()));
	connect(ui->copyrightLabel, SIGNAL(linkActivated(QString)), this, SLOT(openLink(QString)));
}

AboutWindow::~AboutWindow(){
	delete ui;
}

void AboutWindow::openLink(const QString &link){
	QDesktopServices::openUrl(QUrl(link));
}

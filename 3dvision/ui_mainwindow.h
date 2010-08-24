/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created: Sun 20. Jun 19:18:35 2010
**      by: Qt User Interface Compiler version 4.6.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDockWidget>
#include <QtGui/QFrame>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QPlainTextEdit>
#include <QtGui/QPushButton>
#include <QtGui/QStatusBar>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>
#include "UI/cybervisionviewer.h"

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionShowlog;
    QWidget *centralWidget;
    QGridLayout *gridLayout;
    QFrame *frame;
    QHBoxLayout *horizontalLayout;
    QPushButton *startProcessButton;
    QPushButton *saveButton;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout_2;
    CybervisionViewer *openGLViewport;
    QStatusBar *statusBar;
    QDockWidget *logDockWidget;
    QWidget *dockWidgetContents;
    QHBoxLayout *horizontalLayout_2;
    QPlainTextEdit *logTextEdit;
    QMenuBar *menuBar;
    QMenu *menuOptions;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(611, 476);
        actionShowlog = new QAction(MainWindow);
        actionShowlog->setObjectName(QString::fromUtf8("actionShowlog"));
        actionShowlog->setCheckable(true);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        gridLayout = new QGridLayout(centralWidget);
        gridLayout->setSpacing(0);
        gridLayout->setContentsMargins(0, 0, 0, 0);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        frame = new QFrame(centralWidget);
        frame->setObjectName(QString::fromUtf8("frame"));
        QSizePolicy sizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(frame->sizePolicy().hasHeightForWidth());
        frame->setSizePolicy(sizePolicy);
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        horizontalLayout = new QHBoxLayout(frame);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        startProcessButton = new QPushButton(frame);
        startProcessButton->setObjectName(QString::fromUtf8("startProcessButton"));

        horizontalLayout->addWidget(startProcessButton);

        saveButton = new QPushButton(frame);
        saveButton->setObjectName(QString::fromUtf8("saveButton"));
        saveButton->setEnabled(false);

        horizontalLayout->addWidget(saveButton);


        gridLayout->addWidget(frame, 0, 0, 1, 1);

        groupBox = new QGroupBox(centralWidget);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        QSizePolicy sizePolicy1(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(groupBox->sizePolicy().hasHeightForWidth());
        groupBox->setSizePolicy(sizePolicy1);
        verticalLayout_2 = new QVBoxLayout(groupBox);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        openGLViewport = new CybervisionViewer(groupBox);
        openGLViewport->setObjectName(QString::fromUtf8("openGLViewport"));
        sizePolicy1.setHeightForWidth(openGLViewport->sizePolicy().hasHeightForWidth());
        openGLViewport->setSizePolicy(sizePolicy1);

        verticalLayout_2->addWidget(openGLViewport);


        gridLayout->addWidget(groupBox, 1, 0, 1, 1);

        MainWindow->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        MainWindow->setStatusBar(statusBar);
        logDockWidget = new QDockWidget(MainWindow);
        logDockWidget->setObjectName(QString::fromUtf8("logDockWidget"));
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QString::fromUtf8("dockWidgetContents"));
        horizontalLayout_2 = new QHBoxLayout(dockWidgetContents);
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        logTextEdit = new QPlainTextEdit(dockWidgetContents);
        logTextEdit->setObjectName(QString::fromUtf8("logTextEdit"));
        QSizePolicy sizePolicy2(QSizePolicy::Expanding, QSizePolicy::Ignored);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(logTextEdit->sizePolicy().hasHeightForWidth());
        logTextEdit->setSizePolicy(sizePolicy2);
        logTextEdit->setMinimumSize(QSize(0, 50));
        logTextEdit->setUndoRedoEnabled(false);
        logTextEdit->setReadOnly(true);

        horizontalLayout_2->addWidget(logTextEdit);

        logDockWidget->setWidget(dockWidgetContents);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(8), logDockWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 611, 21));
        menuOptions = new QMenu(menuBar);
        menuOptions->setObjectName(QString::fromUtf8("menuOptions"));
        MainWindow->setMenuBar(menuBar);

        menuBar->addAction(menuOptions->menuAction());
        menuOptions->addAction(actionShowlog);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Cybervision", 0, QApplication::UnicodeUTF8));
        actionShowlog->setText(QApplication::translate("MainWindow", "Show log", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        actionShowlog->setToolTip(QApplication::translate("MainWindow", "Show the log", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        startProcessButton->setText(QApplication::translate("MainWindow", "Open && display result", 0, QApplication::UnicodeUTF8));
        saveButton->setText(QApplication::translate("MainWindow", "Save result image", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("MainWindow", "3D image", 0, QApplication::UnicodeUTF8));
        logDockWidget->setWindowTitle(QApplication::translate("MainWindow", "Log", 0, QApplication::UnicodeUTF8));
        menuOptions->setTitle(QApplication::translate("MainWindow", "Options", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H

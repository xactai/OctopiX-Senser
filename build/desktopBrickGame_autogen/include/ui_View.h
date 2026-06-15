/********************************************************************************
** Form generated from reading UI file 'View.ui'
**
** Created by: Qt User Interface Compiler version 5.15.18
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_VIEW_H
#define UI_VIEW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStackedWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_View
{
public:
    QStackedWidget *stackedWidget;
    QWidget *Menu;
    QPushButton *playAgain;
    QPushButton *closeGame;
    QWidget *verticalLayoutWidget;
    QVBoxLayout *verticalLayout;
    QLabel *GameStatus;
    QGridLayout *gridLayout;
    QLabel *label_2;
    QLabel *MenuCurrScore;
    QLabel *label_4;
    QLabel *MenuBestScore;
    QWidget *Entrance;
    QLabel *label_11;
    QPushButton *start_tetris_btn;
    QPushButton *exit_btn;
    QWidget *TetrisGame;
    QWidget *gridLayoutWidget_3;
    QGridLayout *gridLayout_3;
    QLabel *label_5;
    QLabel *label_7;
    QLabel *label_8;
    QLabel *tetris_curr_level;
    QLabel *tetris_curr_score;
    QLabel *tetris_best_score;
    QLabel *label_12;
    QWidget *TetrisGameField;
    QLabel *TetrisInfoLabel;
    QWidget *TetrisGameField_2;
    QLabel *label_13;

    void setupUi(QWidget *View)
    {
        if (View->objectName().isEmpty())
            View->setObjectName(QString::fromUtf8("View"));
        View->resize(501, 602);
        stackedWidget = new QStackedWidget(View);
        stackedWidget->setObjectName(QString::fromUtf8("stackedWidget"));
        stackedWidget->setGeometry(QRect(0, 0, 500, 601));
        stackedWidget->setStyleSheet(QString::fromUtf8(""));
        Menu = new QWidget();
        Menu->setObjectName(QString::fromUtf8("Menu"));
        Menu->setStyleSheet(QString::fromUtf8("QWidget {\n"
"	color: black;\n"
"}"));
        playAgain = new QPushButton(Menu);
        playAgain->setObjectName(QString::fromUtf8("playAgain"));
        playAgain->setGeometry(QRect(50, 360, 400, 81));
        QFont font;
        font.setPointSize(25);
        playAgain->setFont(font);
        closeGame = new QPushButton(Menu);
        closeGame->setObjectName(QString::fromUtf8("closeGame"));
        closeGame->setGeometry(QRect(50, 460, 400, 81));
        closeGame->setFont(font);
        verticalLayoutWidget = new QWidget(Menu);
        verticalLayoutWidget->setObjectName(QString::fromUtf8("verticalLayoutWidget"));
        verticalLayoutWidget->setGeometry(QRect(50, 50, 401, 291));
        verticalLayout = new QVBoxLayout(verticalLayoutWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        GameStatus = new QLabel(verticalLayoutWidget);
        GameStatus->setObjectName(QString::fromUtf8("GameStatus"));
        QFont font1;
        font1.setPointSize(26);
        font1.setBold(true);
        font1.setWeight(75);
        GameStatus->setFont(font1);
        GameStatus->setAlignment(Qt::AlignCenter);

        verticalLayout->addWidget(GameStatus);

        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label_2 = new QLabel(verticalLayoutWidget);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setFont(font);
        label_2->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(label_2, 0, 0, 1, 1);

        MenuCurrScore = new QLabel(verticalLayoutWidget);
        MenuCurrScore->setObjectName(QString::fromUtf8("MenuCurrScore"));
        MenuCurrScore->setFont(font);
        MenuCurrScore->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(MenuCurrScore, 0, 1, 1, 1);

        label_4 = new QLabel(verticalLayoutWidget);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setFont(font);
        label_4->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(label_4, 1, 0, 1, 1);

        MenuBestScore = new QLabel(verticalLayoutWidget);
        MenuBestScore->setObjectName(QString::fromUtf8("MenuBestScore"));
        MenuBestScore->setFont(font);
        MenuBestScore->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(MenuBestScore, 1, 1, 1, 1);


        verticalLayout->addLayout(gridLayout);

        stackedWidget->addWidget(Menu);
        Entrance = new QWidget();
        Entrance->setObjectName(QString::fromUtf8("Entrance"));
        label_11 = new QLabel(Entrance);
        label_11->setObjectName(QString::fromUtf8("label_11"));
        label_11->setGeometry(QRect(60, 10, 400, 131));
        QFont font2;
        font2.setPointSize(42);
        label_11->setFont(font2);
        label_11->setAlignment(Qt::AlignCenter);
        start_tetris_btn = new QPushButton(Entrance);
        start_tetris_btn->setObjectName(QString::fromUtf8("start_tetris_btn"));
        start_tetris_btn->setGeometry(QRect(130, 200, 250, 101));
        QFont font3;
        font3.setPointSize(24);
        start_tetris_btn->setFont(font3);
        exit_btn = new QPushButton(Entrance);
        exit_btn->setObjectName(QString::fromUtf8("exit_btn"));
        exit_btn->setGeometry(QRect(130, 400, 250, 101));
        exit_btn->setFont(font3);
        stackedWidget->addWidget(Entrance);
        TetrisGame = new QWidget();
        TetrisGame->setObjectName(QString::fromUtf8("TetrisGame"));
        gridLayoutWidget_3 = new QWidget(TetrisGame);
        gridLayoutWidget_3->setObjectName(QString::fromUtf8("gridLayoutWidget_3"));
        gridLayoutWidget_3->setGeometry(QRect(310, 250, 191, 281));
        gridLayout_3 = new QGridLayout(gridLayoutWidget_3);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        gridLayout_3->setContentsMargins(0, 0, 0, 0);
        label_5 = new QLabel(gridLayoutWidget_3);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        QFont font4;
        font4.setPointSize(14);
        font4.setBold(false);
        font4.setWeight(50);
        label_5->setFont(font4);
        label_5->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        gridLayout_3->addWidget(label_5, 0, 0, 1, 1);

        label_7 = new QLabel(gridLayoutWidget_3);
        label_7->setObjectName(QString::fromUtf8("label_7"));
        label_7->setFont(font4);
        label_7->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        gridLayout_3->addWidget(label_7, 1, 0, 1, 1);

        label_8 = new QLabel(gridLayoutWidget_3);
        label_8->setObjectName(QString::fromUtf8("label_8"));
        label_8->setFont(font4);
        label_8->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        gridLayout_3->addWidget(label_8, 2, 0, 1, 1);

        tetris_curr_level = new QLabel(gridLayoutWidget_3);
        tetris_curr_level->setObjectName(QString::fromUtf8("tetris_curr_level"));
        QFont font5;
        font5.setPointSize(14);
        tetris_curr_level->setFont(font5);

        gridLayout_3->addWidget(tetris_curr_level, 0, 1, 1, 1);

        tetris_curr_score = new QLabel(gridLayoutWidget_3);
        tetris_curr_score->setObjectName(QString::fromUtf8("tetris_curr_score"));
        tetris_curr_score->setFont(font5);

        gridLayout_3->addWidget(tetris_curr_score, 1, 1, 1, 1);

        tetris_best_score = new QLabel(gridLayoutWidget_3);
        tetris_best_score->setObjectName(QString::fromUtf8("tetris_best_score"));
        tetris_best_score->setFont(font5);

        gridLayout_3->addWidget(tetris_best_score, 2, 1, 1, 1);

        gridLayout_3->setColumnStretch(0, 4);
        gridLayout_3->setColumnStretch(1, 5);
        label_12 = new QLabel(TetrisGame);
        label_12->setObjectName(QString::fromUtf8("label_12"));
        label_12->setGeometry(QRect(310, 10, 191, 51));
        QFont font6;
        font6.setPointSize(26);
        font6.setBold(true);
        font6.setItalic(false);
        font6.setWeight(75);
        label_12->setFont(font6);
        label_12->setAlignment(Qt::AlignCenter);
        TetrisGameField = new QWidget(TetrisGame);
        TetrisGameField->setObjectName(QString::fromUtf8("TetrisGameField"));
        TetrisGameField->setGeometry(QRect(0, 0, 302, 602));
        TetrisGameField->setStyleSheet(QString::fromUtf8("QWidget {\n"
"    border: 2px solid black;\n"
"}"));
        TetrisInfoLabel = new QLabel(TetrisGameField);
        TetrisInfoLabel->setObjectName(QString::fromUtf8("TetrisInfoLabel"));
        TetrisInfoLabel->setGeometry(QRect(20, 210, 261, 201));
        TetrisInfoLabel->setFont(font5);
        TetrisInfoLabel->setAlignment(Qt::AlignCenter);
        TetrisGameField_2 = new QWidget(TetrisGame);
        TetrisGameField_2->setObjectName(QString::fromUtf8("TetrisGameField_2"));
        TetrisGameField_2->setGeometry(QRect(330, 60, 150, 150));
        TetrisGameField_2->setStyleSheet(QString::fromUtf8("QWidget {\n"
"    border: 2px solid black;\n"
"}"));
        label_13 = new QLabel(TetrisGame);
        label_13->setObjectName(QString::fromUtf8("label_13"));
        label_13->setGeometry(QRect(340, 210, 131, 21));
        QFont font7;
        font7.setPointSize(14);
        font7.setBold(true);
        font7.setItalic(false);
        font7.setWeight(75);
        label_13->setFont(font7);
        label_13->setAlignment(Qt::AlignCenter);
        stackedWidget->addWidget(TetrisGame);

        retranslateUi(View);

        stackedWidget->setCurrentIndex(1);


        QMetaObject::connectSlotsByName(View);
    } // setupUi

    void retranslateUi(QWidget *View)
    {
        View->setWindowTitle(QCoreApplication::translate("View", "View", nullptr));
        playAgain->setText(QCoreApplication::translate("View", "Play Again", nullptr));
        closeGame->setText(QCoreApplication::translate("View", "Exit", nullptr));
        GameStatus->setText(QCoreApplication::translate("View", "Game Status", nullptr));
        label_2->setText(QCoreApplication::translate("View", "Your score", nullptr));
        MenuCurrScore->setText(QString());
        label_4->setText(QCoreApplication::translate("View", "Best score", nullptr));
        MenuBestScore->setText(QString());
        label_11->setText(QCoreApplication::translate("View", "Brick Games", nullptr));
        start_tetris_btn->setText(QCoreApplication::translate("View", "Tetris", nullptr));
        exit_btn->setText(QCoreApplication::translate("View", "Exit", nullptr));
        label_5->setText(QCoreApplication::translate("View", "Level", nullptr));
        label_7->setText(QCoreApplication::translate("View", "Score", nullptr));
        label_8->setText(QCoreApplication::translate("View", "Best score", nullptr));
        tetris_curr_level->setText(QString());
        tetris_curr_score->setText(QString());
        tetris_best_score->setText(QString());
        label_12->setText(QCoreApplication::translate("View", "Tetris", nullptr));
        TetrisInfoLabel->setText(QCoreApplication::translate("View", "TextLabel", nullptr));
        label_13->setText(QCoreApplication::translate("View", "Next piece", nullptr));
    } // retranslateUi

};

namespace Ui {
    class View: public Ui_View {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VIEW_H

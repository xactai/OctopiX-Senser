#include <QApplication>

#include "View.h"

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  s21::TetrisModel t_model;
  s21::TetrisController t_controller(&t_model);
  s21::View v(&t_controller);
  v.show();
  return QApplication::exec();
}

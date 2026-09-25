//
// Created by Raisin Ibben on 14.03.2024.
//

#include "common/BrickGameConsoleView.h"

int main() {
  s21::TetrisModel t_model;
  s21::TetrisController t_controller(&t_model);
  s21::BrickGameConsoleView view(&t_controller);
  view.Start();
  return 0;
}

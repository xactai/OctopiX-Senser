
#include "View.h"

#include "ui_View.h"

namespace s21 {

View::View(TetrisController *t_c, QWidget *parent)
    : QMainWindow(parent),
      ui_(new Ui::View),
      current_game_(CurrentGame::kNone),
      action_(UserAction::kNoSig),
      t_data_(nullptr),
      tetris_controller_(t_c) {
  ui_->setupUi(this);
  move(1000, 300);
  setWindowTitle("Tetris");
  ui_->stackedWidget->setCurrentIndex(1);
  tetris_controller_->SetModelDataDefault();
  t_data_ = &tetris_controller_->GetModelData();
  m_timer_ = new QTimer(this);
  connect(m_timer_, &QTimer::timeout, this, &View::UpdateAll);
}

View::~View() {
  if (m_timer_ != nullptr) delete m_timer_;
  delete ui_;
}

void View::keyPressEvent(QKeyEvent *event) {
  int key = event->key();
  switch (key) {
    case Qt::Key_Left:
      action_ = UserAction::kLeft;
      break;
    case Qt::Key_Right:
      action_ = UserAction::kRight;
      break;
    case Qt::Key_Up:
      action_ = UserAction::kUp;
      break;
    case Qt::Key_Down:
      action_ = UserAction::kDown;
      break;
    case Qt::Key_Enter:
      action_ = UserAction::kEnterBtn;
      break;
    case Qt::Key_Tab:
      action_ = UserAction::kTabBtn;
      break;
    case Qt::Key_Space:
      action_ = UserAction::kSpaceBtn;
      break;
    case Qt::Key_Escape:
      action_ = UserAction::kEscBtn;
      break;
    default:
      break;
  }
}

void View::paintEvent(QPaintEvent *event) {
  QWidget::paintEvent(event);
  if (current_game_ == CurrentGame::kTetris) {
    if (t_data_->t_game_status != GameState::kGameOver &&
        t_data_->t_game_status != GameState::kExit) {
      if (t_data_->t_game_status == GameState::kStart) {
        StartWindowRendering(ui_->TetrisInfoLabel);
      } else if (t_data_->t_game_status == GameState::kPause) {
        PauseWindowRendering(ui_->TetrisInfoLabel);
      } else {
        ui_->TetrisInfoLabel->setText("");
        TetrisGameRendering();
      }
    } else {
      ClearField();
      GameOver(false, t_data_->t_level, t_data_->t_score);
    }
  }
}

void View::ClearField() {
  QPainter painter(this);
  painter.eraseRect(rect());
  painter.end();
}

void View::GameOver(bool is_victory, int level, int score) {
  m_timer_->stop();
  ui_->stackedWidget->setCurrentIndex(0);
  if (is_victory) {
    ui_->GameStatus->setText("YOU WON!!!");
  } else {
    ui_->GameStatus->setText("GAME OVER!");
  }
  ui_->MenuCurrScore->setText(QString::number(score));
  ui_->MenuBestScore->setText(QString::number(level));
}

void View::UpdateAll() {
  if (current_game_ == CurrentGame::kTetris) {
    UpdateTetrisModel();
  }
  repaint();
}

void View::on_playAgain_clicked() {
  if (current_game_ == CurrentGame::kTetris) {
    tetris_controller_->SetModelDataDefault();
    t_data_ = &tetris_controller_->GetModelData();
    ui_->stackedWidget->setCurrentIndex(2);
    m_timer_->start(10);
  } else if (current_game_ == CurrentGame::kNone) {
    ui_->stackedWidget->setCurrentIndex(1);
  }
}

void View::on_start_tetris_btn_clicked() {
  current_game_ = CurrentGame::kTetris;
  tetris_controller_->SetModelDataDefault();
  t_data_ = &tetris_controller_->GetModelData();
  ui_->stackedWidget->setCurrentIndex(2);
  m_timer_->start(10);
}

void View::on_exit_btn_clicked() { close(); }

void View::on_closeGame_clicked() {
  current_game_ = CurrentGame::kNone;
  ui_->stackedWidget->setCurrentIndex(1);
}

void View::UpdateTetrisModel() {
  tetris_controller_->UpdateModelData(action_);
  t_data_ = &tetris_controller_->GetModelData();
  action_ = UserAction::kNoSig;
  ui_->tetris_curr_score->setText(QString::number(t_data_->t_score));
  ui_->tetris_curr_level->setText(QString::number(t_data_->t_level));
  ui_->tetris_best_score->setText(QString::number(t_data_->t_best_score));
  if (t_data_->t_game_status == GameState::kGameOver ||
      t_data_->t_game_status == GameState::kExit) {
    m_timer_->stop();
  }
}

void View::TetrisGameRendering() {
  QPainter qp(this);

  qp.setBrush(QColor(90, 90, 90));
  qp.setPen(QColor(0, 0, 0));

  qp.setBrush(kColors[0]);
  for (const auto &item : t_data_->t_projection.GetCoords()) {
    qp.drawRect((item.x) * GameSizes::kDotSize,
                (item.y - 1) * GameSizes::kDotSize, GameSizes::kDotSize - 1,
                GameSizes::kDotSize - 1);
  }

  qp.setBrush(QColor(148, 195, 76));
  for (const auto &item : t_data_->t_curr.GetCoords()) {
    qp.setBrush(kColors[(int)t_data_->t_curr.GetShape()]);
    qp.drawRect(item.x * GameSizes::kDotSize,
                (item.y - 1) * GameSizes::kDotSize, GameSizes::kDotSize - 1,
                GameSizes::kDotSize - 1);
  }

  for (const auto &item : t_data_->t_next.GetCoords()) {
    qp.setBrush(kColors[(int)t_data_->t_next.GetShape()]);
    qp.drawRect((item.x + 8) * GameSizes::kDotSize,
                (item.y + 2) * GameSizes::kDotSize, GameSizes::kDotSize - 1,
                GameSizes::kDotSize - 1);
  }

  for (int i = 0; i < GameSizes::kFieldHeight; ++i) {
    for (int j = 0; j < GameSizes::kFieldWidth; ++j) {
      if (t_data_->t_field_[i][j].first) {
        qp.setBrush(kColors[(int)t_data_->t_field_[i][j].second]);
        qp.drawRect(j * GameSizes::kDotSize, i * GameSizes::kDotSize,
                    GameSizes::kDotSize - 1, GameSizes::kDotSize - 1);
      }
    }
  }
  qp.end();
}

void View::StartWindowRendering(QLabel *p_label) {
  p_label->setText("Press space to start");
  p_label->setStyleSheet("border: none;");
}

void View::PauseWindowRendering(QLabel *p_label) {
  p_label->setText(
      "Game on pause.<br> Press Tab to continue<br> or Esc to exit");
}

}  // namespace s21

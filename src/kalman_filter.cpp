#include "kalman_filter.h"
#include <iostream>
#include <algorithm>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::max;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  // Predict the state
  // Predict mean 
  x_ = F_ * x_; 
  // Predict covariance 
  MatrixXd Ft = F_.transpose(); 
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // Update the state by using Kalman Filter equations

  // KALMAN FILTER UPDATE
  VectorXd y = z - H_ * x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  // KALMAN FILTER PREDICTION
  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size()); // Identity Matrix size is x size by x size
  P_ = (I - K * H_) * P_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // Update the state by using Extended Kalman Filter equations

  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  double rho = sqrt(px*px+py*py);
  double theta = atan2(py,px);
  double rho_dot = (px*vx+py*vy)/max(rho, 0.0001);
  VectorXd z_pred = VectorXd(3);
  z_pred << rho,theta,rho_dot;

  // KALMAN FILTER UPDATE
  VectorXd y = z - z_pred;
  // The range of y(1) should be (-2 * M_PI, 2 * M_PI).
  while (y(1) > 2 * M_PI) 
      y(1) -= 2 * M_PI;
	while (y(1) < -2 * M_PI) 
      y(1) += 2 * M_PI;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  // KALMAN FILTER PREDICTION
  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size()); // Identity Matrix size is x size by x size
  P_ = (I - K * H_) * P_;
}

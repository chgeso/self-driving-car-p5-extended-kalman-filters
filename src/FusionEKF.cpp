#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  // H_laser_ initialization
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // 4x4 matrix (state transition)
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;

  // 4x4 matrix
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0,    0,    0,
             0, 1,    0,    0,
             0, 0, 1000,    0,
             0, 0,    0, 1000;

  // set the acceleration nosie components
  // provided in the quiz as 9 in section 13 of lesson 5.
  noise_ax = 9;
  noise_ay = 9;

}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates 
      // and initialize state.
      
      double rho = measurement_pack.raw_measurements_(0);
      double theta = measurement_pack.raw_measurements_(1);
      double rho_dot = measurement_pack.raw_measurements_(2);

      // The range of theta should be (-2 * M_PI, 2 * M_PI).
      while (theta > 2 * M_PI) 
          theta -= 2 * M_PI;
	    while (theta < -2 * M_PI) 
          theta += 2 * M_PI;
      
      // just set ekf_.x_(0) to rho*cos(theta)
      ekf_.x_(0) = rho*cos(theta);
      // just set ekf_.x_(1) to rho*sin(theta)
      ekf_.x_(1) = rho*sin(theta);
      // just set ekf_.x(2) to rho_dot*cos(theta)
      ekf_.x_(2) = rho_dot*cos(theta);
      // just set ekf_.x(3) to rho_dot*sin(theta)
      ekf_.x_(3) = rho_dot*sin(theta);

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state.
      // just set ekf_.x(0) to x
      ekf_.x_(0) = measurement_pack.raw_measurements_(0);
      // just set ekf_.x(1) to y
      ekf_.x_(1) = measurement_pack.raw_measurements_(1);
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */

  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;  // dt - expressed in seconds.
  previous_timestamp_ = measurement_pack.timestamp_;

  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  // Modify the F matrix so that the time is integrated Section 8 of Lesson 5.
  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;

  // Set the process covariance matrix Q Section 9 of Lesson 5
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4/4*noise_ax,               0, dt_3/2*noise_ax,               0,
                           0, dt_4/4*noise_ay,               0, dt_3/2*noise_ay,
             dt_3/2*noise_ax,               0,   dt_2*noise_ax,               0,
                           0, dt_3/2*noise_ay,               0,   dt_2*noise_ay;

  ekf_.Predict();

  /**
   * Update
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    // Set ekf_.H_ by setting to Hj which is the calculated the jacobian.
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    // Set ekf_.R_ by just using R_radar_
    ekf_.R_ = R_radar_;

    ekf_.UpdateEKF(measurement_pack.raw_measurements_);

  } else {
    // Laser updates
    // Set ekf_.H_ by just using H_laser_
    ekf_.H_ = H_laser_;
    // Set ekf_.R_ by just using R_laser_
    ekf_.R_ = R_laser_;

    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}

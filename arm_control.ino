#include <Adafruit_PWMServoDriver.h>
#include <Wire.h>

// PCA9685 at default I2C address 0x40
Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);

// ── Pulse calibration (PCA9685 tick counts, 0-4095) ──────────────────────────
// At 50Hz: period = 20ms = 20000µs → 1 tick ≈ 4.88µs
//
// MG996R (high-torque): ~500µs (0°) to ~2500µs (180°)
//   500µs  → ~102 ticks
//   2500µs → ~512 ticks
#define MG996R_MIN 102 // ~0°   (500µs pulse)
#define MG996R_MAX 512 // ~180° (2500µs pulse)

// SG90 (micro-servo): ~500µs (0°) to ~2400µs (180°)
//   500µs  → ~102 ticks
//   2400µs → ~491 ticks
#define SG90_MIN 102 // ~0°   (500µs pulse)
#define SG90_MAX 491 // ~180° (2400µs pulse)

// SG90 lower pulse range for Wrist Pitch (ch4) and Gripper (ch6)
//   390µs → ~80 ticks
#define SG90_EXT_MIN  80 // ~0°   (390µs pulse)
#define SG90_EXT_MAX 491 // ~180° (2400µs pulse)

// ── PCA9685 channel assignments ──────────────────────────────────────────────
// MG996R motors (high-torque, 7.4V external supply)
#define CH_BASE 0     // DOF 1 — Base rotation (left/right)
#define CH_SHOULDER 1 // DOF 2 — Shoulder (up/down)
#define CH_ELBOW 3    // DOF 3 — Elbow (index finger curl)

// SG90 micro-servos
#define CH_WRIST_P 4 // DOF 4 — Wrist Pitch (middle finger curl)
#define CH_WRIST_R 5 // DOF 5 — Wrist Roll (thumb spread)
#define CH_GRIPPER 6 // DOF 6 — Gripper (finger count)

// ── Motor type enum ──────────────────────────────────────────────────────────
enum MotorType { MG996R, SG90, SG90_EXT };

// Convert angle (0-180) to PCA9685 pulse tick for the given motor type
uint16_t angleToPulse(int angle, MotorType type) {
  if (type == MG996R) {
    return map(angle, 0, 180, MG996R_MIN, MG996R_MAX);
  } else if (type == SG90_EXT) {
    return map(angle, 0, 180, SG90_EXT_MIN, SG90_EXT_MAX);
  } else {
    return map(angle, 0, 180, SG90_MIN, SG90_MAX);
  }
}

void setServo(uint8_t channel, int angle, MotorType type) {
  angle = constrain(angle, 0, 180);
  pca.setPWM(channel, 0, angleToPulse(angle, type));
}

// Returns the motor type for a given PCA9685 channel
MotorType getMotorType(uint8_t channel) {
  // Base, Shoulder, Elbow = MG996R
  if (channel == CH_BASE || channel == CH_SHOULDER || channel == CH_ELBOW) {
    return MG996R;
  }
  // Wrist Roll = SG90
  if (channel == CH_WRIST_R) {
    return SG90;
  }
  // Wrist Pitch, Gripper = SG90_EXT
  return SG90_EXT;
}

void setServoAuto(uint8_t channel, int angle) {
  setServo(channel, angle, getMotorType(channel));
}

void setup() {
  Serial.begin(9600);
  Wire.begin();

  pca.begin();
  pca.setPWMFreq(50); // 50Hz for standard servos
  delay(10);

  // Move all servos to safe center positions on boot
  setServoAuto(CH_BASE, 90);     // Base center
  setServoAuto(CH_SHOULDER, 90); // Shoulder center
  setServoAuto(CH_ELBOW, 90);    // Elbow center
  setServoAuto(CH_WRIST_P, 90);  // Wrist pitch center
  setServoAuto(CH_WRIST_R, 90);  // Wrist roll center
  setServoAuto(CH_GRIPPER, 50);  // Gripper mid-open

  Serial.println("READY");
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    // Command format: "B,90", "S,120", "E,45", "W,90", "R,90", "G,50"
    if (cmd.length() < 3)
      return;

    char ch_letter = cmd.charAt(0);
    int angle = cmd.substring(2).toInt();
    angle = constrain(angle, 0, 180);

    if (ch_letter == 'B') {
      setServoAuto(CH_BASE, angle);
      Serial.println("OK,B," + String(angle));
    } else if (ch_letter == 'S') {
      setServoAuto(CH_SHOULDER, angle);
      Serial.println("OK,S," + String(angle));
    } else if (ch_letter == 'E') {
      setServoAuto(CH_ELBOW, angle);
      Serial.println("OK,E," + String(angle));
    } else if (ch_letter == 'W') {
      setServoAuto(CH_WRIST_P, angle);
      Serial.println("OK,W," + String(angle));
    } else if (ch_letter == 'R') {
      setServoAuto(CH_WRIST_R, angle);
      Serial.println("OK,R," + String(angle));
    } else if (ch_letter == 'G') {
      setServoAuto(CH_GRIPPER, angle);
      Serial.println("OK,G," + String(angle));
    }
  }
}
// ============================================================
// DRIIFT - Auto-Tracking Camera Mount
// Arduino Uno/Mega main.cpp
// Revised for continuous-rotation servos
// ============================================================

#include <LiquidCrystal.h>
#include <Servo.h>

// ── LCD ──────────────────────────────────────────────────────

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

// ── Continuous-rotation servos ──────────────────────────────

Servo servoX;
Servo servoY;

const int SERVO_X_PIN = 9;
const int SERVO_Y_PIN = 10;

// ── LEDs ─────────────────────────────────────────────────────

const int LED_GREEN = A1;
const int LED_RED   = A2;

// ── Servo Configuration ─────────────────────────────────────
//
// Continuous rotation servos:
// 90 = stop
// <90 = one direction
// >90 = other direction
//
// If servos creep while idle, adjust these values.
// Common values are 88–92.

const int SERVO_X_STOP = 92;
const int SERVO_Y_STOP = 90;

// Maximum offset (pixels) that maps to full speed.
const int FULL_OFFSET_PX = 120;

// Ignore tiny movements near center.
const int DEAD_ZONE = 8;

// Reverse direction if required.
const bool INVERT_X = false;
const bool INVERT_Y = true;

// ── Timeout ──────────────────────────────────────────────────

const unsigned long TIMEOUT_MS = 1500;

// ── State ────────────────────────────────────────────────────

int pwmX = SERVO_X_STOP;
int pwmY = SERVO_Y_STOP;

bool tracking = false;

unsigned long lastMsg = 0;

// LCD refresh throttle
unsigned long lastLCDUpdate = 0;
const unsigned long LCD_INTERVAL = 200;

// ── Helpers ──────────────────────────────────────────────────

void setLED(bool green, bool red)
{
    digitalWrite(LED_GREEN, green ? HIGH : LOW);
    digitalWrite(LED_RED, red ? HIGH : LOW);
}

int offsetToPwm(int offset, int stopVal, bool invert)
{
    if (invert)
    {
        offset = -offset;
    }

    if (abs(offset) < DEAD_ZONE)
    {
        return stopVal;
    }

    int speed = map(
        constrain(abs(offset), 0, FULL_OFFSET_PX),
        0,
        FULL_OFFSET_PX,
        15,
        85
    );

    if (offset < 0)
    {
        speed = -speed;
    }

    return constrain(stopVal + speed, 0, 180);
}

void stopMotors()
{
    pwmX = SERVO_X_STOP;
    pwmY = SERVO_Y_STOP;

    servoX.write(SERVO_X_STOP);
    servoY.write(SERVO_Y_STOP);
}

// ── Setup ────────────────────────────────────────────────────

void setup()
{
    Serial.begin(9600);

    // IMPORTANT:
    // Use standard attach() since your test code works this way.
    servoX.attach(SERVO_X_PIN);
    servoY.attach(SERVO_Y_PIN);

    stopMotors();

    // LEDs
    pinMode(LED_GREEN, OUTPUT);
    pinMode(LED_RED, OUTPUT);

    setLED(false, true);

    // LCD
    lcd.begin(16, 2);
    lcd.clear();

    lcd.setCursor(0, 0);
    lcd.print(" DRIIFT v1.0 ");

    lcd.setCursor(0, 1);
    lcd.print(" Waiting... ");

    delay(1500);

    lcd.clear();

    lcd.setCursor(0, 0);
    lcd.print("Status: IDLE ");

    lcd.setCursor(0, 1);
    lcd.print("No subject     ");

    lastMsg = millis();

    Serial.println("DRIIFT_READY");
}

// ── Main Loop ────────────────────────────────────────────────

void loop()
{
    // --------------------------------------------------------
    // Read serial commands
    // --------------------------------------------------------

    if (Serial.available())
    {
        String raw = Serial.readStringUntil('\n');
        raw.trim();

        if (raw == "LOST")
        {
            tracking = false;
            lastMsg = millis();
        }
        else if (raw == "RESET")
        {
            stopMotors();

            tracking = false;
            lastMsg = millis();

            Serial.println("ACK_RESET");
        }
        else if (raw == "PING")
        {
            Serial.println("PONG");
            lastMsg = millis();
        }
        else
        {
            int commaIdx = raw.indexOf(',');

            if (commaIdx > 0)
            {
                int offX = raw.substring(0, commaIdx).toInt();
                int offY = raw.substring(commaIdx + 1).toInt();

                pwmX = offsetToPwm(
                    offX,
                    SERVO_X_STOP,
                    INVERT_X
                );

                pwmY = offsetToPwm(
                    offY,
                    SERVO_Y_STOP,
                    INVERT_Y
                );

                servoX.write(pwmX);
                servoY.write(pwmY);

                tracking = true;
                lastMsg = millis();

                // Debug output
                Serial.print("X:");
                Serial.print(offX);

                Serial.print(" PWMX:");
                Serial.print(pwmX);

                Serial.print(" Y:");
                Serial.print(offY);

                Serial.print(" PWMY:");
                Serial.println(pwmY);
            }
        }
    }

    // --------------------------------------------------------
    // Timeout detection
    // --------------------------------------------------------

    unsigned long now = millis();

    if (now - lastMsg > TIMEOUT_MS)
    {
        tracking = false;
    }

    // --------------------------------------------------------
    // Stop motors when tracking lost
    // --------------------------------------------------------

    if (!tracking)
    {
        stopMotors();
    }

    // --------------------------------------------------------
    // LEDs
    // --------------------------------------------------------

    setLED(tracking, !tracking);

    // --------------------------------------------------------
    // LCD Updates
    // --------------------------------------------------------

    if (now - lastLCDUpdate >= LCD_INTERVAL)
    {
        lastLCDUpdate = now;

        lcd.setCursor(0, 0);

        if (tracking)
        {
            lcd.print("Status: TRACKING");
        }
        else
        {
            lcd.print("Status: IDLE    ");
        }

        lcd.setCursor(0, 1);

        lcd.print("X:");
        lcd.print(pwmX);

        lcd.print(" Y:");
        lcd.print(pwmY);

        lcd.print("   ");
    }

    delay(10);
}
// ============================================================
// DRIIFT - Auto-Tracking Camera Mount
// Arduino Uno/Mega
// Protocol: LEFT | RIGHT | SAFE | RESET | PING
// ============================================================

#include <LiquidCrystal.h>
#include <Servo.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

Servo servoX;
Servo servoY;

const int SERVO_X_PIN = 9;
const int SERVO_Y_PIN = 10;

const int LED_GREEN = A1;
const int LED_RED   = A2;

// Continuous rotation: 90 = stop, <90 / >90 = spin directions.
const int SERVO_X_STOP = 92;
const int SERVO_Y_STOP = 92;

const int MOVE_SPEED = 3;        // PWM offset when pulsing (tune this: 2–5)
const unsigned long PULSE_DURATION_MS = 40;   // how long each pulse fires (tune this: 40–120ms)
const unsigned long PULSE_COOLDOWN_MS = 90;  // mandatory stop gap between pulses

unsigned long pulseStartTime = 0;
bool inPulse = false;

// Trim if one direction is weaker than the other.
const int SERVO_X_LEFT_TRIM = 4;
const int SERVO_X_RIGHT_TRIM = 0;


const bool INVERT_X = false;
const bool INVERT_Y = false;

const unsigned long TIMEOUT_MS = 1500;

enum Command { CMD_SAFE, CMD_LEFT, CMD_RIGHT };

int pwmX = SERVO_X_STOP;
int pwmY = SERVO_Y_STOP;
int targetPwmX = SERVO_X_STOP;
int targetPwmY = SERVO_Y_STOP;

Command activeCmd = CMD_SAFE;
unsigned long lastMsg = 0;

unsigned long lastLCDUpdate = 0;
const unsigned long LCD_INTERVAL = 200;

void setLED(bool green, bool red)
{
    digitalWrite(LED_GREEN, green ? HIGH : LOW);
    digitalWrite(LED_RED, red ? HIGH : LOW);
}

int pwmForCommand(Command cmd)
{
    if (cmd == CMD_SAFE)
    {
        return SERVO_X_STOP;
    }

    int stop = SERVO_X_STOP;
    int speed = MOVE_SPEED;

    if (cmd == CMD_LEFT)
    {
        stop -= SERVO_X_LEFT_TRIM;
        speed = -MOVE_SPEED;
    }
    else
    {
        stop += SERVO_X_RIGHT_TRIM;
        speed = MOVE_SPEED;
    }

    if (INVERT_X)
    {
        speed = -speed;
    }

    return constrain(stop + speed, 0, 180);
}

void stopMotors(bool immediate = false)
{
    activeCmd = CMD_SAFE;
    targetPwmX = SERVO_X_STOP;
    targetPwmY = SERVO_Y_STOP;

    if (immediate)
    {
        pwmX = SERVO_X_STOP;
        pwmY = SERVO_Y_STOP;
        servoX.write(SERVO_X_STOP);
        servoY.write(SERVO_Y_STOP);
    }
}

void applyCommand(Command cmd)
{
    activeCmd = cmd;
    lastMsg = millis();

    // Only start a new pulse if we're not mid-cooldown
    if (!inPulse) {
        pulseStartTime = millis();
        inPulse = true;
        targetPwmX = pwmForCommand(cmd);
    }
    targetPwmY = SERVO_Y_STOP;
}

const char* commandLabel(Command cmd)
{
    switch (cmd)
    {
        case CMD_LEFT:  return "LEFT ";
        case CMD_RIGHT: return "RIGHT";
        default:        return "SAFE ";
    }
}

void setup()
{
    Serial.begin(115200);

    servoX.attach(SERVO_X_PIN);
    servoY.attach(SERVO_Y_PIN);
    stopMotors(true);

    pinMode(LED_GREEN, OUTPUT);
    pinMode(LED_RED, OUTPUT);
    setLED(false, true);

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
    lcd.print("Cmd: SAFE       ");

    lastMsg = millis();
    Serial.println("DRIIFT_READY");
}

void loop()
{
    if (Serial.available())
    {
        String raw = Serial.readStringUntil('\n');
        raw.trim();

        if (raw == "SAFE" || raw == "LOST")
        {
            stopMotors(true);
            inPulse = false;
        }
        else if (raw == "LEFT")
        {
            applyCommand(CMD_LEFT);
        }
        else if (raw == "RIGHT")
        {
            applyCommand(CMD_RIGHT);
        }
        else if (raw == "RESET")
        {
            stopMotors(true);
            inPulse = false;
            lastMsg = millis();
            Serial.println("ACK_RESET");
        }
        else if (raw == "PING")
        {
            Serial.println("PONG");
            lastMsg = millis();
        }
    }

    unsigned long now = millis();

    // Watchdog: no command received recently
    if (now - lastMsg > TIMEOUT_MS)
    {
        stopMotors(true);
        inPulse = false;
    }

    // Pulse-and-stop logic
    if (inPulse)
    {
        if (now - pulseStartTime >= PULSE_DURATION_MS)
        {
            inPulse = false;
            pwmX = SERVO_X_STOP;
            pwmY = SERVO_Y_STOP;
            servoX.write(SERVO_X_STOP);
            servoY.write(SERVO_Y_STOP);
        }
        else
        {
            servoX.write(targetPwmX);
            servoY.write(targetPwmY);
        }
    }

    setLED(activeCmd != CMD_SAFE, activeCmd == CMD_SAFE);

    if (now - lastLCDUpdate >= LCD_INTERVAL)
    {
        lastLCDUpdate = now;

        lcd.setCursor(0, 0);
        lcd.print(activeCmd == CMD_SAFE ? "Status: IDLE    " : "Status: TRACKING");

        lcd.setCursor(0, 1);
        lcd.print("Cmd:");
        lcd.print(commandLabel(activeCmd));
        lcd.print(" X:");
        lcd.print(pwmX);
        lcd.print("   ");
    }

    delay(10);
}
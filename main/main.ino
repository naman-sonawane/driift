// ============================================================
// DRIIFT - Auto-Tracking Camera Mount
// Arduino Uno/Mega
// Protocol: X:LEFT|RIGHT|SAFE  Y:UP|DOWN|SAFE  RESET  PING
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

const int MOVE_SPEED = 4;        // PWM offset when pulsing (tune this: 4–10)
const unsigned long PULSE_DURATION_MS = 40;   // how long each pulse fires (tune this: 40–120ms)
const unsigned long PULSE_COOLDOWN_MS = 90;  // mandatory stop gap between pulses

unsigned long pulseStartTimeX = 0;
unsigned long pulseEndTimeX = 0;
bool inPulseX = false;

unsigned long pulseStartTimeY = 0;
unsigned long pulseEndTimeY = 0;
bool inPulseY = false;

// Extra delta beyond MOVE_SPEED — tune if one direction is still weak.
const int SERVO_X_LEFT_EXTRA  = 2;
const int SERVO_X_RIGHT_EXTRA = 2;
const int SERVO_Y_UP_EXTRA    = 2;
const int SERVO_Y_DOWN_EXTRA  = 2;

const bool INVERT_X = false;
const bool INVERT_Y = false;

const unsigned long TIMEOUT_MS = 1500;

enum XCommand { X_SAFE, X_LEFT, X_RIGHT };
enum YCommand { Y_SAFE, Y_UP, Y_DOWN };

int pwmX = SERVO_X_STOP;
int pwmY = SERVO_Y_STOP;
int targetPwmX = SERVO_X_STOP;
int targetPwmY = SERVO_Y_STOP;

XCommand activeXCmd = X_SAFE;
YCommand activeYCmd = Y_SAFE;
unsigned long lastMsg = 0;

unsigned long lastLCDUpdate = 0;
const unsigned long LCD_INTERVAL = 200;

void setLED(bool green, bool red)
{
    digitalWrite(LED_GREEN, green ? HIGH : LOW);
    digitalWrite(LED_RED, red ? HIGH : LOW);
}

int pwmForXCommand(XCommand cmd)
{
    if (cmd == X_SAFE)
    {
        return SERVO_X_STOP;
    }

    int delta = MOVE_SPEED;
    if (cmd == X_LEFT)
    {
        delta += SERVO_X_LEFT_EXTRA;
        delta = -delta;
    }
    else
    {
        delta += SERVO_X_RIGHT_EXTRA;
    }

    if (INVERT_X)
    {
        delta = -delta;
    }

    return constrain(SERVO_X_STOP + delta, 0, 180);
}

int pwmForYCommand(YCommand cmd)
{
    if (cmd == Y_SAFE)
    {
        return SERVO_Y_STOP;
    }

    int delta = MOVE_SPEED;
    if (cmd == Y_UP)
    {
        delta += SERVO_Y_UP_EXTRA;
        delta = -delta;
    }
    else
    {
        delta += SERVO_Y_DOWN_EXTRA;
    }

    if (INVERT_Y)
    {
        delta = -delta;
    }

    return constrain(SERVO_Y_STOP + delta, 0, 180);
}

void stopMotors(bool immediate = false)
{
    activeXCmd = X_SAFE;
    activeYCmd = Y_SAFE;
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

void applyXCommand(XCommand cmd)
{
    activeXCmd = cmd;
    lastMsg = millis();

    unsigned long now = millis();
    bool coolingDown = pulseEndTimeX > 0 && (now - pulseEndTimeX) < PULSE_COOLDOWN_MS;

    if (!inPulseX && !coolingDown)
    {
        pulseStartTimeX = now;
        inPulseX = true;
        targetPwmX = pwmForXCommand(cmd);
    }
}

void applyYCommand(YCommand cmd)
{
    activeYCmd = cmd;
    lastMsg = millis();

    unsigned long now = millis();
    bool coolingDown = pulseEndTimeY > 0 && (now - pulseEndTimeY) < PULSE_COOLDOWN_MS;

    if (!inPulseY && !coolingDown)
    {
        pulseStartTimeY = now;
        inPulseY = true;
        targetPwmY = pwmForYCommand(cmd);
    }
}

XCommand parseXCommand(const String& value)
{
    if (value == "LEFT")
    {
        return X_LEFT;
    }
    if (value == "RIGHT")
    {
        return X_RIGHT;
    }
    return X_SAFE;
}

YCommand parseYCommand(const String& value)
{
    if (value == "UP")
    {
        return Y_UP;
    }
    if (value == "DOWN")
    {
        return Y_DOWN;
    }
    return Y_SAFE;
}

void handleAxisCommand(const String& raw)
{
    int colonIdx = raw.indexOf(':');
    if (colonIdx <= 0)
    {
        return;
    }

    String axis = raw.substring(0, colonIdx);
    String value = raw.substring(colonIdx + 1);
    value.trim();

    if (axis == "X")
    {
        if (value == "SAFE")
        {
            activeXCmd = X_SAFE;
            targetPwmX = SERVO_X_STOP;
            inPulseX = false;
            pulseEndTimeX = 0;
            pwmX = SERVO_X_STOP;
            servoX.write(SERVO_X_STOP);
        }
        else
        {
            applyXCommand(parseXCommand(value));
        }
        return;
    }

    if (axis == "Y")
    {
        if (value == "SAFE")
        {
            activeYCmd = Y_SAFE;
            targetPwmY = SERVO_Y_STOP;
            inPulseY = false;
            pulseEndTimeY = 0;
            pwmY = SERVO_Y_STOP;
            servoY.write(SERVO_Y_STOP);
        }
        else
        {
            applyYCommand(parseYCommand(value));
        }
    }
}

const char* xCommandLabel(XCommand cmd)
{
    switch (cmd)
    {
        case X_LEFT:  return "L";
        case X_RIGHT: return "R";
        default:      return "-";
    }
}

const char* yCommandLabel(YCommand cmd)
{
    switch (cmd)
    {
        case Y_UP:    return "U";
        case Y_DOWN:  return "D";
        default:      return "-";
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
            inPulseX = false;
            inPulseY = false;
            pulseEndTimeX = 0;
            pulseEndTimeY = 0;
        }
        else if (raw == "LEFT")
        {
            applyXCommand(X_LEFT);
        }
        else if (raw == "RIGHT")
        {
            applyXCommand(X_RIGHT);
        }
        else if (raw == "UP")
        {
            applyYCommand(Y_UP);
        }
        else if (raw == "DOWN")
        {
            applyYCommand(Y_DOWN);
        }
        else if (raw.startsWith("X:") || raw.startsWith("Y:"))
        {
            handleAxisCommand(raw);
        }
        else if (raw == "RESET")
        {
            stopMotors(true);
            inPulseX = false;
            inPulseY = false;
            pulseEndTimeX = 0;
            pulseEndTimeY = 0;
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
        inPulseX = false;
        inPulseY = false;
        pulseEndTimeX = 0;
        pulseEndTimeY = 0;
    }

    // Pulse-and-stop logic (independent per axis)
    if (inPulseX)
    {
        if (now - pulseStartTimeX >= PULSE_DURATION_MS)
        {
            inPulseX = false;
            pulseEndTimeX = now;
            pwmX = SERVO_X_STOP;
            servoX.write(SERVO_X_STOP);
        }
        else
        {
            pwmX = targetPwmX;
            servoX.write(targetPwmX);
        }
    }

    if (inPulseY)
    {
        if (now - pulseStartTimeY >= PULSE_DURATION_MS)
        {
            inPulseY = false;
            pulseEndTimeY = now;
            pwmY = SERVO_Y_STOP;
            servoY.write(SERVO_Y_STOP);
        }
        else
        {
            pwmY = targetPwmY;
            servoY.write(targetPwmY);
        }
    }

    bool tracking = activeXCmd != X_SAFE || activeYCmd != Y_SAFE || inPulseX || inPulseY;
    setLED(tracking, !tracking);

    if (now - lastLCDUpdate >= LCD_INTERVAL)
    {
        lastLCDUpdate = now;

        lcd.setCursor(0, 0);
        lcd.print(tracking ? "Status: TRACKING" : "Status: IDLE    ");

        lcd.setCursor(0, 1);
        lcd.print("X:");
        lcd.print(xCommandLabel(activeXCmd));
        lcd.print(" Y:");
        lcd.print(yCommandLabel(activeYCmd));
        lcd.print(" ");
        lcd.print(pwmX);
        lcd.print("/");
        lcd.print(pwmY);
        lcd.print(" ");
    }

    delay(10);
}

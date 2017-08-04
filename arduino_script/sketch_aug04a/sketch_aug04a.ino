
const int xpin = A5;                  // x-axis of the accelerometer
const int ypin = A4;                  // y-axis
const int zpin = A3;                  // z-axis (only on 3-axis models)

void setup()
{
  analogReference(EXTERNAL);    //connect 3.3v to AREF
  
  pinMode(A3, INPUT);
  pinMode(A4, INPUT);
  pinMode(A5, INPUT);
  
  Serial.begin(9600);
}

void loop()
{
  // print the sensor values:
  Serial.print(analogRead(xpin) - 508);

  Serial.print(",");
  Serial.print(analogRead(ypin) - 497);

  Serial.print(",");
  Serial.print(analogRead(zpin) - 617);
  Serial.print("\n");

  // Note: this affect the "sample rate"! delay 50ms means 20Hz, 10ms means 93Hz. 
  //       It will miss something when using higher sample rate.
  delay(50);
}

//+------------------------------------------------------------------+
//|                                         NewsPriceDataExporter.mq5 |
//|               Minimal H1/OHLCV exporter for macro news research  |
//+------------------------------------------------------------------+
#property strict
#property version   "1.00"

input int InpHistoryBars = 150000;                 // Maximum historical bars to scan
input string InpTargetSymbol = "";                 // Leave empty to use chart symbol
input bool InpAutoFileName = true;                  // Auto-generate as <SYMBOL>_<TF>_Data.csv
input string InpFileName = "";                      // Custom output file name
input bool InpUseCommonFolder = true;                // Save to Terminal Common Files
input datetime InpStartTime = D'2024.01.01 00:00';  // Export start time
input datetime InpEndTime = D'2026.08.31 23:59';    // Export end time

string gTargetSymbol = "";
string gOutputFileName = "";

string ResolveTargetSymbol()
  {
   string symbol = InpTargetSymbol;
   StringTrimLeft(symbol);
   StringTrimRight(symbol);
   if(StringLen(symbol) == 0)
      return _Symbol;
   StringToUpper(symbol);
   return symbol;
  }

string TimeframeToLabel(const ENUM_TIMEFRAMES timeframe)
  {
   switch(timeframe)
     {
      case PERIOD_M1:  return "M1";
      case PERIOD_M5:  return "M5";
      case PERIOD_M15: return "M15";
      case PERIOD_M30: return "M30";
      case PERIOD_H1:  return "H1";
      case PERIOD_H4:  return "H4";
      case PERIOD_D1:  return "D1";
      case PERIOD_W1:  return "W1";
      case PERIOD_MN1: return "MN1";
      default:         return IntegerToString((int)timeframe);
     }
  }

string ResolveOutputFileName(const string symbol)
  {
   string fileName = InpFileName;
   StringTrimLeft(fileName);
   StringTrimRight(fileName);
   if(InpAutoFileName || StringLen(fileName) == 0)
      return symbol + "_" + TimeframeToLabel(_Period) + "_Data.csv";
   return fileName;
  }

int OnInit()
  {
   gTargetSymbol = ResolveTargetSymbol();
   if(!SymbolSelect(gTargetSymbol, true))
     {
      Print("Failed to select target symbol: ", gTargetSymbol, ". Err: ", GetLastError());
      return INIT_FAILED;
     }

   gOutputFileName = ResolveOutputFileName(gTargetSymbol);
   ExportPriceData();
   ExpertRemove();
   return INIT_SUCCEEDED;
  }

void ExportPriceData()
  {
   uint flags = FILE_WRITE | FILE_CSV | FILE_ANSI;
   if(InpUseCommonFolder)
      flags |= FILE_COMMON;

   int fileHandle = FileOpen(gOutputFileName, flags, ',');
   if(fileHandle == INVALID_HANDLE)
     {
      Print("Error opening file: ", gOutputFileName, ". Err: ", GetLastError());
      return;
     }

   FileWrite(fileHandle, "Time", "Open", "High", "Low", "Close", "Volume", "Hour", "DayOfWeek");

   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   int copied = CopyRates(gTargetSymbol, _Period, 0, InpHistoryBars, rates);
   if(copied <= 0)
     {
      Print("Failed to copy rates for ", gTargetSymbol, ". Err: ", GetLastError());
      FileClose(fileHandle);
      return;
     }

   int targetDigits = (int)SymbolInfoInteger(gTargetSymbol, SYMBOL_DIGITS);
   int exportedRows = 0;

   for(int i = copied - 1; i >= 0; i--)
     {
      if(rates[i].time < InpStartTime || rates[i].time > InpEndTime)
         continue;

      MqlDateTime dt;
      TimeToStruct(rates[i].time, dt);

      FileWrite(fileHandle,
                TimeToString(rates[i].time, TIME_DATE | TIME_MINUTES),
                DoubleToString(rates[i].open, targetDigits),
                DoubleToString(rates[i].high, targetDigits),
                DoubleToString(rates[i].low, targetDigits),
                DoubleToString(rates[i].close, targetDigits),
                IntegerToString(rates[i].tick_volume),
                IntegerToString(dt.hour),
                IntegerToString(dt.day_of_week));
      exportedRows++;
     }

   FileClose(fileHandle);
   Print("NewsPriceDataExporter exported ", exportedRows, " rows to ", gOutputFileName,
         " | symbol=", gTargetSymbol,
         " | timeframe=", TimeframeToLabel(_Period),
         " | common_folder=", (InpUseCommonFolder ? "true" : "false"),
         " | range=", TimeToString(InpStartTime, TIME_DATE | TIME_MINUTES), " to ",
         TimeToString(InpEndTime, TIME_DATE | TIME_MINUTES));
  }

//+------------------------------------------------------------------+
//|                                           NewsEventsTrading.mq5  |
//| Pure macroeconomic news-event trader using CSV event surprises    |
//+------------------------------------------------------------------+
#property strict
#property copyright "Copyright 2026, Abdillah Baradja"
#property link      "dillahbaraja@gmail.com"
#property version   "1.00"

#include <Trade\Trade.mqh>

enum ENUM_LOT_MODE
  {
   LOT_MODE_STATIC,
   LOT_MODE_DYNAMIC_RISK
  };

input string          InpNewsCsvFileName       = "Dataset\\NEWS_EVENTS.csv";
input bool            InpUseCommonFolder       = true;
input ulong           InpMagicNumber           = 20260905;
input ENUM_LOT_MODE   InpLotMode               = LOT_MODE_STATIC;
input double          InpLotSize               = 0.10;
input double          InpRiskPercent           = 2.0;
input int             InpStopLoss              = 200;
input int             InpTakeProfit            = 400;
input int             InpMaxSpread             = 30;
input int             InpMaxOpenPosition       = 1;
input bool            InpAllowReverse          = true;
input bool            InpUsePairCountryFilter  = false;
input int             InpMinImportance         = 0;
input int             InpMinImpact             = 0;
input int             InpTradeWindowMinutes    = 15;
input int             InpTimerSeconds          = 5;
input bool            InpTradeOnlyNewBarInTester = true;
input bool            InpOneTradePerEvent      = true;
input string          InpInverseEventKeywords  = "";
input bool            InpShowPanel             = true;
input int             InpMaxPanelEvents        = 8;
input bool            InpEnableDebugLog        = true;
input bool            InpEnableDebugCsv        = true;
input string          InpDebugCsvFileName      = "NewsEventsTrading_debug.csv";
input bool            InpEnableTransactionCsv  = true;
input bool            InpExportHistoryOnDeinit = true;
input string          InpTransactionCsvFile    = "";
input string          InpEventTradeCsvFile     = "";

CTrade trade;

datetime gNewsEventTimes[];
string   gNewsEventNames[];
string   gNewsCountryCodes[];
int      gNewsImportances[];
int      gNewsImpacts[];
double   gNewsActuals[];
double   gNewsForecasts[];
double   gNewsPreviouss[];
bool     gNewsHasActuals[];
bool     gNewsHasForecasts[];
bool     gNewsHasPreviouss[];

string   gBaseCountryCode = "";
string   gQuoteCountryCode = "";
string   gTradedEventKeys[];
datetime gLastNewsLoadTime = 0;
datetime gLastBarTime = 0;
bool     gNewsLoaded = false;
string   gLastDecision = "None";
string   gLastTradeStatus = "None";
string   gTransactionCsvFile = "";
string   gEventTradeCsvFile = "";

int OpenCsv(const string file_name, const int flags);
bool LoadNewsEvents();
bool IsRelevantCountryCode(const string country_code);
string ResolveCountryCode(const string currency_code);
string NormalizeCountryCodeText(const string country_code);
string NormalizeCsvText(const string text);
datetime BuildCsvDateTime(const string date_text, const string time_text);
string SanitizeCsvText(const string value);
bool IsInverseFundamentalEvent(const string event_name);
bool StringMatchesDelimitedKeyword(const string text, const string keyword_list);
bool ResolveEventSignal(const int index, int &signal, string &reason);
bool ProcessNewsWindow();
bool ProcessNewsWindowAt(const datetime evaluation_time);
bool ExecuteSignal(const int signal,
                   const string order_comment,
                   double &lot_size,
                   double &open_price,
                   double &sl,
                   double &tp,
                   ulong &order_ticket,
                   ulong &position_id);
double CalculateOrderLotSize();
int CountOpenPositions(int &buy_count, int &sell_count);
void ClosePositions(const ENUM_POSITION_TYPE type);
string ResolveEventKey(const int index);
bool HasTradedEventKey(const string event_key);
void RememberTradedEventKey(const string event_key);
bool AppendDebugCsv(const int index, const int signal, const string status, const ulong retcode=0, const string retcode_text="");
bool AppendEventTradeCsv(const int index,
                         const int signal,
                         const string order_comment,
                         const double lot_size,
                         const double open_price,
                         const double sl,
                         const double tp,
                         const ulong order_ticket,
                         const ulong position_id,
                         const ulong retcode,
                         const string retcode_text,
                         const string status);
bool AppendDealCsv(const ulong deal_ticket);
bool ExportAllDealHistoryCsv();
void RenderPanel();
void DebugPrint(const string message);
string SignalToText(const int signal);
string ValueToText(const bool has_value, const double value);
bool IsTesterMode();
bool IsNewBar();
string ResolveTransactionCsvFile();
string ResolveEventTradeCsvFile();
string DealTypeToText(const ENUM_DEAL_TYPE deal_type);
string DealEntryToText(const ENUM_DEAL_ENTRY deal_entry);

//+------------------------------------------------------------------+
void DebugPrint(const string message)
  {
   if(InpEnableDebugLog)
      Print(message);
  }

//+------------------------------------------------------------------+
int OpenCsv(const string file_name, const int flags)
  {
   if(InpUseCommonFolder)
      return FileOpen(file_name, flags | FILE_COMMON, ',');
   return FileOpen(file_name, flags, ',');
  }

//+------------------------------------------------------------------+
string NormalizeCsvText(const string text)
  {
   string value = text;
   StringTrimLeft(value);
   StringTrimRight(value);
   StringReplace(value, "\"", "");
   return value;
  }

//+------------------------------------------------------------------+
string SanitizeCsvText(const string value)
  {
   string text = value;
   StringReplace(text, "\r", " ");
   StringReplace(text, "\n", " ");
   StringReplace(text, "\"", "");
   StringReplace(text, ",", " ");
   StringTrimLeft(text);
   StringTrimRight(text);
   return text;
  }

//+------------------------------------------------------------------+
string NormalizeCountryCodeText(const string country_code)
  {
   string code = NormalizeCsvText(country_code);
   StringToUpper(code);
   return code;
  }

//+------------------------------------------------------------------+
datetime BuildCsvDateTime(const string date_text, const string time_text)
  {
   string clean_date = NormalizeCsvText(date_text);
   string clean_time = NormalizeCsvText(time_text);

   if(StringLen(clean_date) >= 19)
      return StringToTime(clean_date);

   if(StringLen(clean_date) >= 10 && StringLen(clean_time) >= 5)
     {
      MqlDateTime dt;
      ZeroMemory(dt);
      dt.year = (int)StringToInteger(StringSubstr(clean_date, 0, 4));
      dt.mon  = (int)StringToInteger(StringSubstr(clean_date, 5, 2));
      dt.day  = (int)StringToInteger(StringSubstr(clean_date, 8, 2));
      dt.hour = (int)StringToInteger(StringSubstr(clean_time, 0, 2));
      dt.min  = (int)StringToInteger(StringSubstr(clean_time, 3, 2));
      dt.sec  = (StringLen(clean_time) >= 8 ? (int)StringToInteger(StringSubstr(clean_time, 6, 2)) : 0);
      return StructToTime(dt);
     }

   return StringToTime(clean_date + " " + clean_time);
  }

//+------------------------------------------------------------------+
string ResolveCountryCode(const string currency_code)
  {
   string cc = currency_code;
   StringToUpper(cc);
   if(cc == "EUR") return "EU";
   if(cc == "USD") return "US";
   if(cc == "GBP") return "GB";
   if(cc == "JPY") return "JP";
   if(cc == "AUD") return "AU";
   if(cc == "CAD") return "CA";
   if(cc == "NZD") return "NZ";
   if(cc == "CHF") return "CH";
   if(cc == "XAU") return "US";
   if(cc == "XTI") return "US";
   return "";
  }

//+------------------------------------------------------------------+
bool IsRelevantCountryCode(const string country_code)
  {
   string code = NormalizeCountryCodeText(country_code);
   if(!InpUsePairCountryFilter)
      return true;
   return (code == gBaseCountryCode || code == gQuoteCountryCode);
  }

//+------------------------------------------------------------------+
bool LoadNewsEvents()
  {
   int handle = OpenCsv(InpNewsCsvFileName, FILE_READ | FILE_CSV | FILE_ANSI);
   if(handle == INVALID_HANDLE)
     {
      Print("Failed to open news CSV: ", InpNewsCsvFileName, " Err: ", GetLastError());
      return false;
     }

   ArrayResize(gNewsEventTimes, 0);
   ArrayResize(gNewsEventNames, 0);
   ArrayResize(gNewsCountryCodes, 0);
   ArrayResize(gNewsImportances, 0);
   ArrayResize(gNewsImpacts, 0);
   ArrayResize(gNewsActuals, 0);
   ArrayResize(gNewsForecasts, 0);
   ArrayResize(gNewsPreviouss, 0);
   ArrayResize(gNewsHasActuals, 0);
   ArrayResize(gNewsHasForecasts, 0);
   ArrayResize(gNewsHasPreviouss, 0);

   string header_first = "";
   if(!FileIsEnding(handle))
      header_first = NormalizeCsvText(FileReadString(handle));
   bool has_id_column = (header_first == "Id");
   while(!FileIsLineEnding(handle) && !FileIsEnding(handle))
      FileReadString(handle);

   int count = 0;
   while(!FileIsEnding(handle))
     {
      string first_value = FileReadString(handle);
      if(StringLen(first_value) == 0 && FileIsEnding(handle))
         break;

      string datetime_value = (has_id_column ? FileReadString(handle) : first_value);
      string time_value = FileReadString(handle);
      string name_value = FileReadString(handle);
      string country_code = FileReadString(handle);
      string country_name = FileReadString(handle);
      string importance_value = FileReadString(handle);
      string actual_value = NormalizeCsvText(FileReadString(handle));
      string forecast_value = NormalizeCsvText(FileReadString(handle));
      string previous_value = NormalizeCsvText(FileReadString(handle));
      string impact_value = FileReadString(handle);
      string url_value = FileReadString(handle);

      datetime event_time = BuildCsvDateTime(datetime_value, time_value);
      if(event_time > 0)
        {
         ArrayResize(gNewsEventTimes, count + 1);
         ArrayResize(gNewsEventNames, count + 1);
         ArrayResize(gNewsCountryCodes, count + 1);
         ArrayResize(gNewsImportances, count + 1);
         ArrayResize(gNewsImpacts, count + 1);
         ArrayResize(gNewsActuals, count + 1);
         ArrayResize(gNewsForecasts, count + 1);
         ArrayResize(gNewsPreviouss, count + 1);
         ArrayResize(gNewsHasActuals, count + 1);
         ArrayResize(gNewsHasForecasts, count + 1);
         ArrayResize(gNewsHasPreviouss, count + 1);

         gNewsEventTimes[count] = event_time;
         gNewsEventNames[count] = SanitizeCsvText(name_value);
         gNewsCountryCodes[count] = NormalizeCountryCodeText(country_code);
         gNewsImportances[count] = (int)StringToInteger(importance_value);
         gNewsImpacts[count] = (int)StringToInteger(impact_value);
         gNewsHasActuals[count] = (StringLen(actual_value) > 0);
         gNewsHasForecasts[count] = (StringLen(forecast_value) > 0);
         gNewsHasPreviouss[count] = (StringLen(previous_value) > 0);
         gNewsActuals[count] = (gNewsHasActuals[count] ? StringToDouble(actual_value) : 0.0);
         gNewsForecasts[count] = (gNewsHasForecasts[count] ? StringToDouble(forecast_value) : 0.0);
         gNewsPreviouss[count] = (gNewsHasPreviouss[count] ? StringToDouble(previous_value) : 0.0);
         count++;
        }

      while(!FileIsLineEnding(handle) && !FileIsEnding(handle))
         FileReadString(handle);
     }

   FileClose(handle);
   gLastNewsLoadTime = TimeCurrent();
   gNewsLoaded = true;
   Print("NewsEventsTrading loaded ", count, " rows from ", InpNewsCsvFileName);
   return (count > 0);
  }

//+------------------------------------------------------------------+
bool StringMatchesDelimitedKeyword(const string text, const string keyword_list)
  {
   string target = text;
   string keywords = keyword_list;
   StringToLower(target);
   StringToLower(keywords);

   string parts[];
   int total = StringSplit(keywords, ';', parts);
   for(int i = 0; i < total; i++)
     {
      string keyword = parts[i];
      StringTrimLeft(keyword);
      StringTrimRight(keyword);
      if(StringLen(keyword) > 0 && StringFind(target, keyword) >= 0)
         return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
bool IsInverseFundamentalEvent(const string event_name)
  {
   return StringMatchesDelimitedKeyword(event_name, InpInverseEventKeywords);
  }

//+------------------------------------------------------------------+
bool ResolveEventSignal(const int index, int &signal, string &reason)
  {
   signal = 0;
   reason = "";

   if(index < 0 || index >= ArraySize(gNewsEventTimes))
     {
      reason = "invalid_event_index";
      return false;
     }
   if(!gNewsHasActuals[index])
     {
      reason = "missing_actual";
      return false;
     }
   if(InpUsePairCountryFilter && !IsRelevantCountryCode(gNewsCountryCodes[index]))
     {
      reason = "country_not_relevant";
      return false;
     }
   if(gNewsImportances[index] < InpMinImportance)
     {
      reason = "importance_below_min";
      return false;
     }
   if(gNewsImpacts[index] < InpMinImpact)
     {
      reason = "impact_below_min";
      return false;
     }

   int forecast_cmp = 0;
   int previous_cmp = 0;
   double actual = gNewsActuals[index];

   if(gNewsHasForecasts[index])
     {
      if(actual > gNewsForecasts[index])
         forecast_cmp = 1;
      else if(actual < gNewsForecasts[index])
         forecast_cmp = -1;
     }

   if(gNewsHasPreviouss[index])
     {
      if(actual > gNewsPreviouss[index])
         previous_cmp = 1;
      else if(actual < gNewsPreviouss[index])
         previous_cmp = -1;
     }

   int macro_direction = 0;
   if(forecast_cmp != 0 && previous_cmp != 0)
     {
      if(forecast_cmp != previous_cmp)
        {
         reason = "forecast_previous_conflict";
         return false;
        }
      macro_direction = forecast_cmp;
     }
   else if(forecast_cmp != 0)
      macro_direction = forecast_cmp;
   else if(previous_cmp != 0)
      macro_direction = previous_cmp;
   else
     {
      reason = "no_surprise";
      return false;
     }

   if(IsInverseFundamentalEvent(gNewsEventNames[index]))
      macro_direction = -macro_direction;

   bool bullish_for_currency = (macro_direction > 0);
   string event_country = NormalizeCountryCodeText(gNewsCountryCodes[index]);
   if(event_country == gBaseCountryCode)
      signal = (bullish_for_currency ? 1 : -1);
   else if(event_country == gQuoteCountryCode)
      signal = (bullish_for_currency ? -1 : 1);
   else if(!InpUsePairCountryFilter)
      signal = (bullish_for_currency ? 1 : -1);
   else
     {
      reason = "country_not_relevant";
      return false;
     }

   reason = "signal_ready";
   return (signal != 0);
  }

//+------------------------------------------------------------------+
string ResolveEventKey(const int index)
  {
   if(index < 0 || index >= ArraySize(gNewsEventTimes))
      return "";
   return TimeToString(gNewsEventTimes[index], TIME_DATE | TIME_SECONDS) + "|" +
          gNewsCountryCodes[index] + "|" + gNewsEventNames[index];
  }

//+------------------------------------------------------------------+
bool HasTradedEventKey(const string event_key)
  {
   for(int i = 0; i < ArraySize(gTradedEventKeys); i++)
      if(gTradedEventKeys[i] == event_key)
         return true;
   return false;
  }

//+------------------------------------------------------------------+
void RememberTradedEventKey(const string event_key)
  {
   if(StringLen(event_key) == 0 || HasTradedEventKey(event_key))
      return;
   int count = ArraySize(gTradedEventKeys);
   ArrayResize(gTradedEventKeys, count + 1);
   gTradedEventKeys[count] = event_key;
  }

//+------------------------------------------------------------------+
string SignalToText(const int signal)
  {
   if(signal > 0)
      return "BUY";
   if(signal < 0)
      return "SELL";
   return "NONE";
  }

//+------------------------------------------------------------------+
string ValueToText(const bool has_value, const double value)
  {
   if(!has_value)
      return "";
   return DoubleToString(value, 6);
  }

//+------------------------------------------------------------------+
bool IsTesterMode()
  {
   return (MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_OPTIMIZATION));
  }

//+------------------------------------------------------------------+
bool IsNewBar()
  {
   datetime current_time = iTime(_Symbol, _Period, 0);
   if(current_time != gLastBarTime)
     {
      gLastBarTime = current_time;
      return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
string ResolveTransactionCsvFile()
  {
   if(StringLen(InpTransactionCsvFile) > 0)
      return InpTransactionCsvFile;
   return "NewsEventsTrading_" + _Symbol + "_transactions.csv";
  }

//+------------------------------------------------------------------+
string ResolveEventTradeCsvFile()
  {
   if(StringLen(InpEventTradeCsvFile) > 0)
      return InpEventTradeCsvFile;
   return "NewsEventsTrading_" + _Symbol + "_event_trades.csv";
  }

//+------------------------------------------------------------------+
string DealTypeToText(const ENUM_DEAL_TYPE deal_type)
  {
   if(deal_type == DEAL_TYPE_BUY) return "BUY";
   if(deal_type == DEAL_TYPE_SELL) return "SELL";
   if(deal_type == DEAL_TYPE_BALANCE) return "BALANCE";
   if(deal_type == DEAL_TYPE_CREDIT) return "CREDIT";
   if(deal_type == DEAL_TYPE_CHARGE) return "CHARGE";
   if(deal_type == DEAL_TYPE_CORRECTION) return "CORRECTION";
   if(deal_type == DEAL_TYPE_BONUS) return "BONUS";
   if(deal_type == DEAL_TYPE_COMMISSION) return "COMMISSION";
   if(deal_type == DEAL_TYPE_COMMISSION_DAILY) return "COMMISSION_DAILY";
   if(deal_type == DEAL_TYPE_COMMISSION_MONTHLY) return "COMMISSION_MONTHLY";
   if(deal_type == DEAL_TYPE_COMMISSION_AGENT_DAILY) return "COMMISSION_AGENT_DAILY";
   if(deal_type == DEAL_TYPE_COMMISSION_AGENT_MONTHLY) return "COMMISSION_AGENT_MONTHLY";
   if(deal_type == DEAL_TYPE_INTEREST) return "INTEREST";
   if(deal_type == DEAL_TYPE_BUY_CANCELED) return "BUY_CANCELED";
   if(deal_type == DEAL_TYPE_SELL_CANCELED) return "SELL_CANCELED";
   if(deal_type == DEAL_DIVIDEND) return "DIVIDEND";
   if(deal_type == DEAL_DIVIDEND_FRANKED) return "DIVIDEND_FRANKED";
   if(deal_type == DEAL_TAX) return "TAX";
   return "OTHER";
  }

//+------------------------------------------------------------------+
string DealEntryToText(const ENUM_DEAL_ENTRY deal_entry)
  {
   if(deal_entry == DEAL_ENTRY_IN) return "IN";
   if(deal_entry == DEAL_ENTRY_OUT) return "OUT";
   if(deal_entry == DEAL_ENTRY_INOUT) return "INOUT";
   if(deal_entry == DEAL_ENTRY_OUT_BY) return "OUT_BY";
   return "OTHER";
  }

//+------------------------------------------------------------------+
bool AppendDebugCsv(const int index, const int signal, const string status, const ulong retcode=0, const string retcode_text="")
  {
   if(!InpEnableDebugCsv)
      return true;

   bool file_exists = (InpUseCommonFolder ? FileIsExist(InpDebugCsvFileName, FILE_COMMON) : FileIsExist(InpDebugCsvFileName));
   int handle = OpenCsv(InpDebugCsvFileName, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI);
   if(handle == INVALID_HANDLE)
     {
      Print("Failed to open debug CSV: ", InpDebugCsvFileName, " Err: ", GetLastError());
      return false;
     }

   if(!file_exists || FileSize(handle) <= 0)
     {
      FileWrite(handle,
                "logged_at",
                "server_time",
                "event_time",
                "symbol",
                "country",
                "event_name",
                "importance",
                "impact",
                "actual",
                "forecast",
                "previous",
                "signal",
                "spread",
                "status",
                "retcode",
                "retcode_text");
     }
   else
      FileSeek(handle, 0, SEEK_END);

   string event_time = "";
   string country = "";
   string event_name = "";
   string importance = "";
   string impact = "";
   string actual = "";
   string forecast = "";
   string previous = "";

   if(index >= 0 && index < ArraySize(gNewsEventTimes))
     {
      event_time = TimeToString(gNewsEventTimes[index], TIME_DATE | TIME_SECONDS);
      country = gNewsCountryCodes[index];
      event_name = gNewsEventNames[index];
      importance = IntegerToString(gNewsImportances[index]);
      impact = IntegerToString(gNewsImpacts[index]);
      actual = ValueToText(gNewsHasActuals[index], gNewsActuals[index]);
      forecast = ValueToText(gNewsHasForecasts[index], gNewsForecasts[index]);
      previous = ValueToText(gNewsHasPreviouss[index], gNewsPreviouss[index]);
     }

   FileWrite(handle,
             TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
             TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
             event_time,
             _Symbol,
             country,
             event_name,
             importance,
             impact,
             actual,
             forecast,
             previous,
             SignalToText(signal),
             IntegerToString((int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD)),
             SanitizeCsvText(status),
             IntegerToString((int)retcode),
             SanitizeCsvText(retcode_text));
   FileClose(handle);
   return true;
  }

//+------------------------------------------------------------------+
bool AppendEventTradeCsv(const int index,
                         const int signal,
                         const string order_comment,
                         const double lot_size,
                         const double open_price,
                         const double sl,
                         const double tp,
                         const ulong order_ticket,
                         const ulong position_id,
                         const ulong retcode,
                         const string retcode_text,
                         const string status)
  {
   if(!InpEnableTransactionCsv)
      return true;

   bool file_exists = (InpUseCommonFolder ? FileIsExist(gEventTradeCsvFile, FILE_COMMON) : FileIsExist(gEventTradeCsvFile));
   int handle = OpenCsv(gEventTradeCsvFile, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI);
   if(handle == INVALID_HANDLE)
     {
      Print("Failed to open event trade CSV: ", gEventTradeCsvFile, " Err: ", GetLastError());
      return false;
     }

   if(!file_exists || FileSize(handle) <= 0)
     {
      FileWrite(handle,
                "SERVER_TIME",
                "EVENT_TIME",
                "SYMBOL",
                "EVENT_COUNTRY",
                "EVENT_NAME_FULL",
                "ORDER_COMMENT",
                "IMPORTANCE",
                "IMPACT",
                "ACTUAL",
                "FORECAST",
                "PREVIOUS",
                "SIGNAL",
                "LOT",
                "OPEN_PRICE",
                "SL",
                "TP",
                "SPREAD",
                "ORDER_TICKET",
                "POSITION_ID",
                "RETCODE",
                "RETCODE_TEXT",
                "STATUS");
     }
   else
      FileSeek(handle, 0, SEEK_END);

   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   FileWrite(handle,
             TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
             (index >= 0 && index < ArraySize(gNewsEventTimes) ? TimeToString(gNewsEventTimes[index], TIME_DATE | TIME_SECONDS) : ""),
             _Symbol,
             (index >= 0 && index < ArraySize(gNewsCountryCodes) ? gNewsCountryCodes[index] : ""),
             (index >= 0 && index < ArraySize(gNewsEventNames) ? gNewsEventNames[index] : ""),
             SanitizeCsvText(order_comment),
             (index >= 0 && index < ArraySize(gNewsImportances) ? IntegerToString(gNewsImportances[index]) : ""),
             (index >= 0 && index < ArraySize(gNewsImpacts) ? IntegerToString(gNewsImpacts[index]) : ""),
             (index >= 0 && index < ArraySize(gNewsActuals) ? ValueToText(gNewsHasActuals[index], gNewsActuals[index]) : ""),
             (index >= 0 && index < ArraySize(gNewsForecasts) ? ValueToText(gNewsHasForecasts[index], gNewsForecasts[index]) : ""),
             (index >= 0 && index < ArraySize(gNewsPreviouss) ? ValueToText(gNewsHasPreviouss[index], gNewsPreviouss[index]) : ""),
             SignalToText(signal),
             DoubleToString(lot_size, 2),
             DoubleToString(open_price, digits),
             DoubleToString(sl, digits),
             DoubleToString(tp, digits),
             IntegerToString((int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD)),
             (string)order_ticket,
             (string)position_id,
             (string)retcode,
             SanitizeCsvText(retcode_text),
             SanitizeCsvText(status));
   FileClose(handle);
   return true;
  }

//+------------------------------------------------------------------+
bool AppendDealCsv(const ulong deal_ticket)
  {
   if(!InpEnableTransactionCsv || deal_ticket == 0)
      return false;
   if(!HistoryDealSelect(deal_ticket))
      return false;

   string symbol = HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
   if(symbol != _Symbol)
      return true;
   if((ulong)HistoryDealGetInteger(deal_ticket, DEAL_MAGIC) != InpMagicNumber)
      return true;

   bool file_exists = (InpUseCommonFolder ? FileIsExist(gTransactionCsvFile, FILE_COMMON) : FileIsExist(gTransactionCsvFile));
   int handle = OpenCsv(gTransactionCsvFile, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI);
   if(handle == INVALID_HANDLE)
     {
      Print("Failed to open transaction CSV: ", gTransactionCsvFile, " Err: ", GetLastError());
      return false;
     }

   if(!file_exists || FileSize(handle) <= 0)
     {
      FileWrite(handle,
                "TIME",
                "DEAL_TICKET",
                "ORDER_TICKET",
                "POSITION_ID",
                "SYMBOL",
                "DEAL_TYPE",
                "ENTRY",
                "VOLUME",
                "PRICE",
                "SL",
                "TP",
                "PROFIT",
                "SWAP",
                "COMMISSION",
                "COMMENT");
     }
   else
      FileSeek(handle, 0, SEEK_END);

   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   FileWrite(handle,
             TimeToString((datetime)HistoryDealGetInteger(deal_ticket, DEAL_TIME), TIME_DATE | TIME_SECONDS),
             (string)deal_ticket,
             (string)HistoryDealGetInteger(deal_ticket, DEAL_ORDER),
             (string)HistoryDealGetInteger(deal_ticket, DEAL_POSITION_ID),
             symbol,
             DealTypeToText((ENUM_DEAL_TYPE)HistoryDealGetInteger(deal_ticket, DEAL_TYPE)),
             DealEntryToText((ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal_ticket, DEAL_ENTRY)),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_VOLUME), 2),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_PRICE), digits),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_SL), digits),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_TP), digits),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_PROFIT), 2),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_SWAP), 2),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION), 2),
             SanitizeCsvText(HistoryDealGetString(deal_ticket, DEAL_COMMENT)));
   FileClose(handle);
   return true;
  }

//+------------------------------------------------------------------+
bool ExportAllDealHistoryCsv()
  {
   if(!InpEnableTransactionCsv)
      return false;
   if(!HistorySelect(0, TimeCurrent()))
      return false;

   int handle = OpenCsv(gTransactionCsvFile, FILE_WRITE | FILE_CSV | FILE_ANSI);
   if(handle == INVALID_HANDLE)
     {
      Print("Failed to rewrite transaction CSV: ", gTransactionCsvFile, " Err: ", GetLastError());
      return false;
     }

   FileWrite(handle,
             "TIME",
             "DEAL_TICKET",
             "ORDER_TICKET",
             "POSITION_ID",
             "SYMBOL",
             "DEAL_TYPE",
             "ENTRY",
             "VOLUME",
             "PRICE",
             "SL",
             "TP",
             "PROFIT",
             "SWAP",
             "COMMISSION",
             "COMMENT");

   int total = HistoryDealsTotal();
   for(int i = 0; i < total; i++)
     {
      ulong deal_ticket = HistoryDealGetTicket(i);
      if(deal_ticket == 0 || !HistoryDealSelect(deal_ticket))
         continue;

      string symbol = HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
      if(symbol != _Symbol)
         continue;
      if((ulong)HistoryDealGetInteger(deal_ticket, DEAL_MAGIC) != InpMagicNumber)
         continue;

      int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
      FileWrite(handle,
                TimeToString((datetime)HistoryDealGetInteger(deal_ticket, DEAL_TIME), TIME_DATE | TIME_SECONDS),
                (string)deal_ticket,
                (string)HistoryDealGetInteger(deal_ticket, DEAL_ORDER),
                (string)HistoryDealGetInteger(deal_ticket, DEAL_POSITION_ID),
                symbol,
                DealTypeToText((ENUM_DEAL_TYPE)HistoryDealGetInteger(deal_ticket, DEAL_TYPE)),
                DealEntryToText((ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal_ticket, DEAL_ENTRY)),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_VOLUME), 2),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_PRICE), digits),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_SL), digits),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_TP), digits),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_PROFIT), 2),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_SWAP), 2),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION), 2),
                SanitizeCsvText(HistoryDealGetString(deal_ticket, DEAL_COMMENT)));
     }

   FileClose(handle);
   return true;
  }

//+------------------------------------------------------------------+
double CalculateOrderLotSize()
  {
   double min_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double volume_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   if(min_volume <= 0.0)
      min_volume = 0.01;
   if(max_volume < min_volume)
      max_volume = min_volume;
   if(volume_step <= 0.0)
      volume_step = min_volume;

   double raw_lot = InpLotSize;
   if(InpLotMode == LOT_MODE_DYNAMIC_RISK)
     {
      double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      double balance = AccountInfoDouble(ACCOUNT_BALANCE);

      if(InpStopLoss > 0 && point > 0.0 && tick_size > 0.0 && tick_value > 0.0 && balance > 0.0 && InpRiskPercent > 0.0)
        {
         double risk_amount = balance * (InpRiskPercent / 100.0);
         double stop_distance = InpStopLoss * point;
         double ticks_to_stop = stop_distance / tick_size;
         double risk_per_lot = ticks_to_stop * tick_value;
         if(risk_per_lot > 0.0)
            raw_lot = risk_amount / risk_per_lot;
        }
     }

   double steps = MathFloor(raw_lot / volume_step);
   double normalized_lot = steps * volume_step;
   if(normalized_lot < min_volume)
      normalized_lot = min_volume;
   if(normalized_lot > max_volume)
      normalized_lot = max_volume;

   int volume_digits = 2;
   if(volume_step > 0.0)
      volume_digits = (int)MathMax(0.0, MathRound(-MathLog10(volume_step)));

   return NormalizeDouble(normalized_lot, volume_digits);
  }

//+------------------------------------------------------------------+
int CountOpenPositions(int &buy_count, int &sell_count)
  {
   buy_count = 0;
   sell_count = 0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;

      long type = PositionGetInteger(POSITION_TYPE);
      if(type == POSITION_TYPE_BUY)
         buy_count++;
      else if(type == POSITION_TYPE_SELL)
         sell_count++;
     }

   return buy_count + sell_count;
  }

//+------------------------------------------------------------------+
void ClosePositions(const ENUM_POSITION_TYPE type)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      if(PositionGetInteger(POSITION_TYPE) == type)
         trade.PositionClose(ticket);
     }
  }

//+------------------------------------------------------------------+
bool ExecuteSignal(const int signal,
                   const string order_comment,
                   double &lot_size,
                   double &open_price,
                   double &sl,
                   double &tp,
                   ulong &order_ticket,
                   ulong &position_id)
  {
   lot_size = 0.0;
   open_price = 0.0;
   sl = 0.0;
   tp = 0.0;
   order_ticket = 0;
   position_id = 0;

   if(signal == 0)
      return false;

   int buy_count = 0;
   int sell_count = 0;
   int total_positions = CountOpenPositions(buy_count, sell_count);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   lot_size = CalculateOrderLotSize();

   if(signal > 0)
     {
      if(sell_count > 0 && InpAllowReverse)
        {
         ClosePositions(POSITION_TYPE_SELL);
         total_positions = CountOpenPositions(buy_count, sell_count);
        }
      if(total_positions >= InpMaxOpenPosition || buy_count > 0)
         return false;

      open_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      sl = (InpStopLoss > 0 ? open_price - InpStopLoss * point : 0.0);
      tp = (InpTakeProfit > 0 ? open_price + InpTakeProfit * point : 0.0);
      bool ok = trade.Buy(lot_size, _Symbol, open_price, sl, tp, order_comment);
      order_ticket = trade.ResultOrder();
      if(trade.ResultDeal() > 0 && HistoryDealSelect(trade.ResultDeal()))
         position_id = (ulong)HistoryDealGetInteger(trade.ResultDeal(), DEAL_POSITION_ID);
      if(!ok)
         Print("BUY failed | retcode=", trade.ResultRetcode(), " | ", trade.ResultComment());
      return ok;
     }

   if(buy_count > 0 && InpAllowReverse)
     {
      ClosePositions(POSITION_TYPE_BUY);
      total_positions = CountOpenPositions(buy_count, sell_count);
     }
   if(total_positions >= InpMaxOpenPosition || sell_count > 0)
      return false;

   open_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   sl = (InpStopLoss > 0 ? open_price + InpStopLoss * point : 0.0);
   tp = (InpTakeProfit > 0 ? open_price - InpTakeProfit * point : 0.0);
   bool ok = trade.Sell(lot_size, _Symbol, open_price, sl, tp, order_comment);
   order_ticket = trade.ResultOrder();
   if(trade.ResultDeal() > 0 && HistoryDealSelect(trade.ResultDeal()))
      position_id = (ulong)HistoryDealGetInteger(trade.ResultDeal(), DEAL_POSITION_ID);
   if(!ok)
      Print("SELL failed | retcode=", trade.ResultRetcode(), " | ", trade.ResultComment());
   return ok;
  }

//+------------------------------------------------------------------+
bool ProcessNewsWindow()
  {
   return ProcessNewsWindowAt(TimeCurrent());
  }

//+------------------------------------------------------------------+
bool ProcessNewsWindowAt(const datetime evaluation_time)
  {
   if(!gNewsLoaded && !LoadNewsEvents())
      return false;

   int window_seconds = (IsTesterMode() ? 3600 : MathMax(1, InpTradeWindowMinutes) * 60);
   int spread = (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   bool processed = false;

   for(int i = 0; i < ArraySize(gNewsEventTimes); i++)
     {
      datetime event_time = gNewsEventTimes[i];
      int age_seconds = (int)(evaluation_time - event_time);
      if(age_seconds < 0 || age_seconds > window_seconds)
         continue;

      processed = true;
      int signal = 0;
      string reason = "";
      if(!ResolveEventSignal(i, signal, reason))
        {
         gLastDecision = StringFormat("Skip %s | %s", gNewsEventNames[i], reason);
         AppendDebugCsv(i, 0, reason);
         continue;
        }

      string event_key = ResolveEventKey(i);
      if(InpOneTradePerEvent && HasTradedEventKey(event_key))
        {
         gLastDecision = "Skip " + gNewsEventNames[i] + " | event_already_traded";
         AppendDebugCsv(i, signal, "event_already_traded");
         continue;
        }

      if(InpMaxSpread > 0 && spread > InpMaxSpread)
        {
         gLastDecision = StringFormat("Skip %s | spread_too_high %d>%d", gNewsEventNames[i], spread, InpMaxSpread);
         AppendDebugCsv(i, signal, "spread_too_high");
         continue;
        }

      string comment = gNewsEventNames[i];
      if(StringLen(comment) > 31)
         comment = StringSubstr(comment, 0, 31);

      double lot_size = 0.0;
      double open_price = 0.0;
      double sl = 0.0;
      double tp = 0.0;
      ulong order_ticket = 0;
      ulong position_id = 0;
      bool executed = ExecuteSignal(signal, comment, lot_size, open_price, sl, tp, order_ticket, position_id);
      ulong retcode = trade.ResultRetcode();
      string retcode_text = trade.ResultComment();
      if(executed)
        {
         if(InpOneTradePerEvent)
            RememberTradedEventKey(event_key);
         gLastTradeStatus = StringFormat("%s executed | %s | %s",
                                         SignalToText(signal),
                                         gNewsCountryCodes[i],
                                         gNewsEventNames[i]);
         gLastDecision = "Trade executed";
         AppendDebugCsv(i, signal, "trade_executed", retcode, retcode_text);
         AppendEventTradeCsv(i, signal, comment, lot_size, open_price, sl, tp, order_ticket, position_id, retcode, retcode_text, "trade_executed");
         DebugPrint(gLastTradeStatus);
         return true;
        }

      gLastTradeStatus = StringFormat("%s failed/skipped | retcode=%d | %s",
                                      SignalToText(signal),
                                      (int)retcode,
                                      retcode_text);
      gLastDecision = gLastTradeStatus;
      AppendDebugCsv(i, signal, "execution_failed_or_skipped", retcode, retcode_text);
      AppendEventTradeCsv(i, signal, comment, lot_size, open_price, sl, tp, order_ticket, position_id, retcode, retcode_text, "execution_failed_or_skipped");
     }

   if(!processed)
      gLastDecision = "No event inside trade window";
   return false;
  }

//+------------------------------------------------------------------+
void RenderPanel()
  {
   if(!InpShowPanel || IsTesterMode())
     {
      Comment("");
      return;
     }

   datetime now_time = TimeCurrent();
   string panel = "NewsEventsTrading | " + _Symbol +
                  "\nServer: " + TimeToString(now_time, TIME_DATE | TIME_SECONDS) +
                  "\nCSV: " + InpNewsCsvFileName +
                  "\nRows: " + IntegerToString(ArraySize(gNewsEventTimes)) +
                  "\nPair country filter: " + (InpUsePairCountryFilter ? gBaseCountryCode + "/" + gQuoteCountryCode : "OFF") +
                  "\nLast decision: " + gLastDecision +
                  "\nLast trade: " + gLastTradeStatus +
                  "\nUpcoming events:";

   int shown = 0;
   for(int i = 0; i < ArraySize(gNewsEventTimes) && shown < InpMaxPanelEvents; i++)
     {
      if(gNewsEventTimes[i] < now_time)
         continue;
      if(InpUsePairCountryFilter && !IsRelevantCountryCode(gNewsCountryCodes[i]))
         continue;
      if(gNewsImportances[i] < InpMinImportance || gNewsImpacts[i] < InpMinImpact)
         continue;

      panel += StringFormat("\n%s %s %s I%d M%d",
                            TimeToString(gNewsEventTimes[i], TIME_DATE | TIME_MINUTES),
                            gNewsCountryCodes[i],
                            gNewsEventNames[i],
                            gNewsImportances[i],
                            gNewsImpacts[i]);
      shown++;
     }

   if(shown == 0)
      panel += "\n(none)";
   Comment(panel);
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(InpMagicNumber);
   gBaseCountryCode = ResolveCountryCode(StringSubstr(_Symbol, 0, 3));
   gQuoteCountryCode = ResolveCountryCode(StringSubstr(_Symbol, 3, 3));
   gTransactionCsvFile = ResolveTransactionCsvFile();
   gEventTradeCsvFile = ResolveEventTradeCsvFile();
   ArrayResize(gTradedEventKeys, 0);

   if(StringLen(gBaseCountryCode) == 0 || StringLen(gQuoteCountryCode) == 0)
     {
      Print("Unsupported symbol country mapping: ", _Symbol);
      return INIT_FAILED;
     }

   if(!LoadNewsEvents())
      return INIT_FAILED;

   EventSetTimer(MathMax(1, InpTimerSeconds));
   Print("NewsEventsTrading initialized | symbol=", _Symbol,
         " | country_filter=", (InpUsePairCountryFilter ? gBaseCountryCode + "/" + gQuoteCountryCode : "OFF"),
         " | file=", InpNewsCsvFileName,
         " | event_trade_csv=", gEventTradeCsvFile,
         " | transaction_csv=", gTransactionCsvFile,
         " | min_importance=", InpMinImportance,
         " | min_impact=", InpMinImpact,
         " | window_minutes=", InpTradeWindowMinutes);
   RenderPanel();
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   Comment("");
   EventKillTimer();
   if(InpExportHistoryOnDeinit)
      ExportAllDealHistoryCsv();
  }

//+------------------------------------------------------------------+
void OnTimer()
  {
   if(IsTesterMode())
      return;

   ProcessNewsWindow();
   if(!IsTesterMode())
      RenderPanel();
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   if(IsTesterMode())
     {
      if(InpTradeOnlyNewBarInTester && !IsNewBar())
         return;

      MqlRates rates[2];
      if(CopyRates(_Symbol, PERIOD_H1, 0, 2, rates) < 2)
         return;

      datetime first_rate_time = rates[0].time;
      datetime second_rate_time = rates[1].time;
      datetime current_bar_time = (first_rate_time > second_rate_time ? first_rate_time : second_rate_time);
      datetime previous_bar_time = (first_rate_time > second_rate_time ? second_rate_time : first_rate_time);
      if(previous_bar_time >= current_bar_time)
         return;

      ProcessNewsWindowAt(current_bar_time);
      return;
     }

   ProcessNewsWindow();
   RenderPanel();
  }

//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
  {
   if(!InpEnableTransactionCsv)
      return;
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD || trans.deal == 0)
      return;

   AppendDealCsv(trans.deal);
  }
//+------------------------------------------------------------------+

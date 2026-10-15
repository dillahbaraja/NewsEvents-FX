//+------------------------------------------------------------------+
//| NewsEventsModelTester.mq5                                        |
//| Strategy Tester runtime for macro-news FX model artifacts.        |
//+------------------------------------------------------------------+
#property strict
#property copyright "Copyright 2026, Abdillah Baradja"
#property version   "1.00"

#include <Trade/Trade.mqh>

enum ENUM_NEWS_MODEL_SELECTION
  {
   NEWS_MODEL_V2_CATBOOST_PAIR_ONNX = 0,
   NEWS_MODEL_V1_CATBOOST_CSV = 1,
   NEWS_MODEL_V1_ELASTIC_NET_CSV = 2,
   NEWS_MODEL_V1_RANDOM_FOREST_CSV = 3,
   NEWS_MODEL_V1_SVM_RBF_CSV = 4,
   NEWS_MODEL_V1_EBM_CSV = 5,
   NEWS_MODEL_V2_SVM_RBF_CSV = 6,
   NEWS_MODEL_V2_EBM_CSV = 7,
   NEWS_MODEL_EVENT_STUDY_V3_MIXED_CSV = 8
  };

enum ENUM_LOT_MODE
  {
   LOT_MODE_FIXED = 0,
   LOT_MODE_DYNAMIC_RISK = 1
  };

enum ENUM_EXIT_MODE
  {
   EXIT_SL_TP = 0,
   EXIT_AFTER_HOURS = 1,
   EXIT_SL_TP_OR_AFTER_HOURS = 2
  };

input ENUM_NEWS_MODEL_SELECTION InpModelSelection   = NEWS_MODEL_V2_CATBOOST_PAIR_ONNX;
input string                  InpNewsCsvFileName    = "Dataset\\datatest\\NEWS_EVENTS_TEST.csv";
input bool                    InpUseCommonFolder    = true;
input ulong                   InpMagicNumber        = 20260911;
input ENUM_LOT_MODE           InpLotMode            = LOT_MODE_FIXED;
input double                  InpFixedLot           = 0.10;
input double                  InpRiskPercent        = 5.0;
input ENUM_EXIT_MODE          InpExitMode           = EXIT_AFTER_HOURS;
input int                     InpHoldingHours       = 1;
input int                     InpStopLossPoints     = 300;
input int                     InpTakeProfitPoints   = 600;
input int                     InpMaxSpreadPoints    = 30;
input int                     InpTradeWindowMinutes = 15;
input bool                    InpJournalRelevantEvents = true;
input bool                    InpJournalPredictionDetails = false;
input bool                    InpOneTradePerEvent   = true;
input bool                    InpAllowReverse       = true;
input bool                    InpExportHistoryOnDeinit = true;
input string                  InpPredictionCsvFile  = "";
input string                  InpTransactionCsvFile = "";
input bool                    InpEnableLog          = true;

CTrade trade;
long gOnnxModel = INVALID_HANDLE;
const double INTERNAL_PROBABILITY_BUY = 0.45;
const double INTERNAL_PROBABILITY_SELL = 0.45;
const bool   INTERNAL_ALLOW_NO_TRADE_OVERRIDE = true;
const double INTERNAL_DIRECTIONAL_OVERRIDE_PROBABILITY = 0.40;
const double INTERNAL_NO_TRADE_DOMINANCE_MARGIN = 0.20;

datetime gEventTimes[];
string   gEventNames[];
string   gEventCountries[];
int      gEventImportances[];
int      gEventImpacts[];
double   gActuals[];
double   gForecasts[];
double   gPrevious[];
bool     gHasActuals[];
bool     gHasForecasts[];
bool     gHasPrevious[];
string   gTradedKeys[];
ulong    gPositionMapIds[];
datetime gPositionMapEventTimes[];
string   gPositionMapEventCountries[];
string   gPositionMapEventNames[];
string   gPositionMapModels[];
datetime gPositionMapOpenTimes[];

string gBaseCurrency = "";
string gQuoteCurrency = "";
string gBaseCountry = "";
string gQuoteCountry = "";
string gPredictionCsv = "";
string gTransactionCsv = "";
string gLastStatus = "";
string gLastMonitoredEvent = "";
datetime gLastJournalBarTime = 0;
int    gJournalScanIndex = 0;
int    gTradeScanIndex = 0;
int    gNextEventScanIndex = 0;
string gEffectiveOnnxModelFile = "";
string gEffectiveCsvSignalFile = "";
string gEffectiveCsvModelName = "";
bool   gUseOnnxRuntime = false;

string gCsvKeys[];
string gCsvSymbols[];
double gCsvProbabilities[];
int    gCsvSignals[];
string gCsvModels[];

//+------------------------------------------------------------------+
int OpenCsv(const string file_name, const int flags)
  {
   if(InpUseCommonFolder)
      return FileOpen(file_name, flags | FILE_COMMON, ',');
   return FileOpen(file_name, flags, ',');
  }

//+------------------------------------------------------------------+
string CleanText(const string value)
  {
   string text = value;
   StringTrimLeft(text);
   StringTrimRight(text);
   StringReplace(text, "\r", " ");
   StringReplace(text, "\n", " ");
   StringReplace(text, "\"", "'");
   return text;
  }

//+------------------------------------------------------------------+
string UpperText(const string value)
  {
   string text = CleanText(value);
   StringToUpper(text);
   return text;
  }

//+------------------------------------------------------------------+
string CsvText(const string value)
  {
   string text = CleanText(value);
   StringReplace(text, ",", " ");
   return text;
  }

//+------------------------------------------------------------------+
string ModelSelectionName()
  {
   switch(InpModelSelection)
     {
      case NEWS_MODEL_V2_CATBOOST_PAIR_ONNX:    return "v2_catboost_pair_onnx";
      case NEWS_MODEL_V1_CATBOOST_CSV:          return "v1_catboost_csv";
      case NEWS_MODEL_V1_ELASTIC_NET_CSV:       return "v1_elastic_net_csv";
      case NEWS_MODEL_V1_RANDOM_FOREST_CSV:     return "v1_random_forest_csv";
      case NEWS_MODEL_V1_SVM_RBF_CSV:           return "v1_svm_rbf_csv";
      case NEWS_MODEL_V1_EBM_CSV:               return "v1_ebm_csv";
      case NEWS_MODEL_V2_SVM_RBF_CSV:           return "v2_svm_rbf_csv";
      case NEWS_MODEL_V2_EBM_CSV:               return "v2_ebm_csv";
      case NEWS_MODEL_EVENT_STUDY_V3_MIXED_CSV: return "event_study_v3_mixed_csv";
     }
   return "unknown";
  }
//+------------------------------------------------------------------+
bool IsOnnxSelection()
  {
   return (InpModelSelection == NEWS_MODEL_V2_CATBOOST_PAIR_ONNX);
  }

//+------------------------------------------------------------------+
bool IsCatBoostV2PairSelection()
  {
   return (InpModelSelection == NEWS_MODEL_V2_CATBOOST_PAIR_ONNX);
  }
//+------------------------------------------------------------------+
bool IsCatBoostPairMulticlass()
  {
   return (InpModelSelection == NEWS_MODEL_V2_CATBOOST_PAIR_ONNX && UpperText(_Symbol) == "AUDUSD");
  }

//+------------------------------------------------------------------+
double CatBoostPairThreshold()
  {
   string symbol = UpperText(_Symbol);
   if(symbol == "GBPUSD" || symbol == "AUDUSD")
      return 0.65;
   if(symbol == "NZDUSD" || symbol == "USDCHF" || symbol == "GBPJPY")
      return 0.75;
   return 0.70;
  }

//+------------------------------------------------------------------+
int CatBoostPairMinImportance()
  {
   return 2;
  }

//+------------------------------------------------------------------+
int CatBoostPairMinImpact()
  {
   string symbol = UpperText(_Symbol);
   if(symbol == "EURUSD" || symbol == "GBPUSD" || symbol == "EURJPY")
      return 1;
   return 2;
  }

//+------------------------------------------------------------------+
bool PassesModelEventFilter(const int index)
  {
   if(!IsCatBoostV2PairSelection())
      return true;
   if(gEventImportances[index] < CatBoostPairMinImportance())
      return false;
   if(gEventImpacts[index] < CatBoostPairMinImpact())
      return false;
   return true;
  }

//+------------------------------------------------------------------+
string ResolveOnnxModelFile()
  {
   if(InpModelSelection == NEWS_MODEL_V2_CATBOOST_PAIR_ONNX)
      return "Dataset\\artifacts\\catboost_v2_pair_onnx\\catboost_v2_pair_" + UpperText(_Symbol) + ".onnx";
   return "";
  }
//+------------------------------------------------------------------+
string ResolveCsvSignalFile()
  {
   switch(InpModelSelection)
     {
      case NEWS_MODEL_EVENT_STUDY_V3_MIXED_CSV:
         return "Dataset\\artifacts\\event_study_v3_signal_table.csv";
      case NEWS_MODEL_V2_SVM_RBF_CSV:
      case NEWS_MODEL_V2_EBM_CSV:
         return "Dataset\\artifacts\\paper1_svm_ebm_v2_mt5_signal_table.csv";
      default:
         return "Dataset\\artifacts\\mt5_signal_table.csv";
     }
  }
//+------------------------------------------------------------------+
string ResolveCsvModelName()
  {
   switch(InpModelSelection)
     {
      case NEWS_MODEL_V1_CATBOOST_CSV:          return "catboost";
      case NEWS_MODEL_V1_ELASTIC_NET_CSV:       return "elastic_net";
      case NEWS_MODEL_V1_RANDOM_FOREST_CSV:     return "random_forest";
      case NEWS_MODEL_V1_SVM_RBF_CSV:           return "svm_rbf";
      case NEWS_MODEL_V1_EBM_CSV:               return "ebm";
      case NEWS_MODEL_V2_SVM_RBF_CSV:           return "svm_rbf_v2";
      case NEWS_MODEL_V2_EBM_CSV:               return "ebm_v2";
      case NEWS_MODEL_EVENT_STUDY_V3_MIXED_CSV: return "event_study_v3";
      default:                                  return "";
     }
  }

//+------------------------------------------------------------------+
datetime ParseEventTime(const string datetime_text, const string time_text)
  {
   string dt = CleanText(datetime_text);
   string tm = CleanText(time_text);
   if(StringLen(dt) >= 19)
      return StringToTime(dt);
   return StringToTime(dt + " " + tm);
  }

//+------------------------------------------------------------------+
bool ParseNumber(const string text, double &value)
  {
   string v = CleanText(text);
   if(StringLen(v) == 0)
      return false;
   StringReplace(v, "%", "");
   StringReplace(v, "K", "");
   StringReplace(v, "M", "");
   StringReplace(v, "B", "");
   StringReplace(v, "T", "");
   StringReplace(v, ",", "");
   value = StringToDouble(v);
   return true;
  }

//+------------------------------------------------------------------+
string CountryToCurrency(const string country)
  {
   string c = UpperText(country);
   if(c == "US") return "USD";
   if(c == "EU") return "EUR";
   if(c == "GB") return "GBP";
   if(c == "JP") return "JPY";
   if(c == "AU") return "AUD";
   if(c == "CA") return "CAD";
   if(c == "NZ") return "NZD";
   if(c == "CH") return "CHF";
   return "";
  }

//+------------------------------------------------------------------+
string CurrencyToCountry(const string currency)
  {
   string c = UpperText(currency);
   if(c == "USD") return "US";
   if(c == "EUR") return "EU";
   if(c == "GBP") return "GB";
   if(c == "JPY") return "JP";
   if(c == "AUD") return "AU";
   if(c == "CAD") return "CA";
   if(c == "NZD") return "NZ";
   if(c == "CHF") return "CH";
   return "";
  }

//+------------------------------------------------------------------+
string EventKey(const int index)
  {
   return TimeToString(gEventTimes[index], TIME_DATE | TIME_MINUTES) + "|" +
          gEventCountries[index] + "|" + gEventNames[index] + "|" + _Symbol;
  }

//+------------------------------------------------------------------+
bool WasTraded(const string key)
  {
   for(int i = 0; i < ArraySize(gTradedKeys); i++)
      if(gTradedKeys[i] == key)
         return true;
   return false;
  }

//+------------------------------------------------------------------+
void MarkTraded(const string key)
  {
   int n = ArraySize(gTradedKeys);
   ArrayResize(gTradedKeys, n + 1);
   gTradedKeys[n] = key;
  }

//+------------------------------------------------------------------+
void RegisterPositionEvent(const ulong position_id, const int event_index)
  {
   if(position_id == 0 || event_index < 0 || event_index >= ArraySize(gEventTimes))
      return;

   for(int i = 0; i < ArraySize(gPositionMapIds); i++)
     {
      if(gPositionMapIds[i] == position_id)
        {
         gPositionMapEventTimes[i] = gEventTimes[event_index];
         gPositionMapEventCountries[i] = gEventCountries[event_index];
         gPositionMapEventNames[i] = gEventNames[event_index];
         gPositionMapModels[i] = ModelSelectionName();
         if(i >= ArraySize(gPositionMapOpenTimes))
            ArrayResize(gPositionMapOpenTimes, ArraySize(gPositionMapIds));
         if(gPositionMapOpenTimes[i] <= 0)
            gPositionMapOpenTimes[i] = TimeCurrent();
         return;
        }
     }

   int n = ArraySize(gPositionMapIds);
   ArrayResize(gPositionMapIds, n + 1);
   ArrayResize(gPositionMapEventTimes, n + 1);
   ArrayResize(gPositionMapEventCountries, n + 1);
   ArrayResize(gPositionMapEventNames, n + 1);
   ArrayResize(gPositionMapModels, n + 1);
   ArrayResize(gPositionMapOpenTimes, n + 1);
   gPositionMapIds[n] = position_id;
   gPositionMapEventTimes[n] = gEventTimes[event_index];
   gPositionMapEventCountries[n] = gEventCountries[event_index];
   gPositionMapEventNames[n] = gEventNames[event_index];
   gPositionMapModels[n] = ModelSelectionName();
   gPositionMapOpenTimes[n] = TimeCurrent();
  }

//+------------------------------------------------------------------+
bool LookupPositionEvent(const ulong position_id,
                         datetime &event_time,
                         string &event_country,
                         string &event_name,
                         string &model_name)
  {
   for(int i = 0; i < ArraySize(gPositionMapIds); i++)
     {
      if(gPositionMapIds[i] == position_id)
        {
         event_time = gPositionMapEventTimes[i];
         event_country = gPositionMapEventCountries[i];
         event_name = gPositionMapEventNames[i];
         model_name = gPositionMapModels[i];
         return true;
        }
     }
   event_time = 0;
   event_country = "";
   event_name = "";
   model_name = "";
   return false;
  }

//+------------------------------------------------------------------+
datetime LookupPositionOpenTime(const ulong position_id)
  {
   for(int i = 0; i < ArraySize(gPositionMapIds); i++)
     {
      if(gPositionMapIds[i] == position_id && i < ArraySize(gPositionMapOpenTimes))
         return gPositionMapOpenTimes[i];
     }
   return 0;
  }

//+------------------------------------------------------------------+
string ResolvePredictionCsv()
  {
   if(StringLen(InpPredictionCsvFile) > 0)
      return InpPredictionCsvFile;
   return "NewsEventsModelTester_" + _Symbol + "_" + ModelSelectionName() + "_predictions.csv";
  }

//+------------------------------------------------------------------+
string ResolveTransactionCsv()
  {
   if(StringLen(InpTransactionCsvFile) > 0)
      return InpTransactionCsvFile;
   return "NewsEventsModelTester_" + _Symbol + "_" + ModelSelectionName() + "_transactions.csv";
  }

//+------------------------------------------------------------------+
void SwapEvents(const int i, const int j)
  {
   datetime time_value = gEventTimes[i];
   gEventTimes[i] = gEventTimes[j];
   gEventTimes[j] = time_value;

   string name_value = gEventNames[i];
   gEventNames[i] = gEventNames[j];
   gEventNames[j] = name_value;

   string country_value = gEventCountries[i];
   gEventCountries[i] = gEventCountries[j];
   gEventCountries[j] = country_value;

   int importance_value = gEventImportances[i];
   gEventImportances[i] = gEventImportances[j];
   gEventImportances[j] = importance_value;

   int impact_value = gEventImpacts[i];
   gEventImpacts[i] = gEventImpacts[j];
   gEventImpacts[j] = impact_value;

   double actual_value = gActuals[i];
   gActuals[i] = gActuals[j];
   gActuals[j] = actual_value;

   double forecast_value = gForecasts[i];
   gForecasts[i] = gForecasts[j];
   gForecasts[j] = forecast_value;

   double previous_value = gPrevious[i];
   gPrevious[i] = gPrevious[j];
   gPrevious[j] = previous_value;

   bool has_actual_value = gHasActuals[i];
   gHasActuals[i] = gHasActuals[j];
   gHasActuals[j] = has_actual_value;

   bool has_forecast_value = gHasForecasts[i];
   gHasForecasts[i] = gHasForecasts[j];
   gHasForecasts[j] = has_forecast_value;

   bool has_previous_value = gHasPrevious[i];
   gHasPrevious[i] = gHasPrevious[j];
   gHasPrevious[j] = has_previous_value;
  }

//+------------------------------------------------------------------+
bool EventLess(const int left, const int right)
  {
   if(gEventTimes[left] != gEventTimes[right])
      return gEventTimes[left] < gEventTimes[right];
   if(gEventCountries[left] != gEventCountries[right])
      return gEventCountries[left] < gEventCountries[right];
   return gEventNames[left] < gEventNames[right];
  }

//+------------------------------------------------------------------+
bool EventLessThanValues(const int index, const datetime pivot_time, const string pivot_country, const string pivot_name)
  {
   if(gEventTimes[index] != pivot_time)
      return gEventTimes[index] < pivot_time;
   if(gEventCountries[index] != pivot_country)
      return gEventCountries[index] < pivot_country;
   return gEventNames[index] < pivot_name;
  }

//+------------------------------------------------------------------+
bool ValuesLessThanEvent(const datetime pivot_time, const string pivot_country, const string pivot_name, const int index)
  {
   if(pivot_time != gEventTimes[index])
      return pivot_time < gEventTimes[index];
   if(pivot_country != gEventCountries[index])
      return pivot_country < gEventCountries[index];
   return pivot_name < gEventNames[index];
  }

//+------------------------------------------------------------------+
void QuickSortEvents(const int left, const int right)
  {
   int i = left;
   int j = right;
   int pivot = left + (right - left) / 2;
   datetime pivot_time = gEventTimes[pivot];
   string pivot_country = gEventCountries[pivot];
   string pivot_name = gEventNames[pivot];

   while(i <= j)
     {
      while(EventLessThanValues(i, pivot_time, pivot_country, pivot_name))
         i++;
      while(ValuesLessThanEvent(pivot_time, pivot_country, pivot_name, j))
         j--;
      if(i <= j)
        {
         SwapEvents(i, j);
         i++;
         j--;
        }
     }

   if(left < j)
      QuickSortEvents(left, j);
   if(i < right)
      QuickSortEvents(i, right);
  }

//+------------------------------------------------------------------+
void SortLoadedNewsEvents()
  {
   int total = ArraySize(gEventTimes);
   if(total > 1)
      QuickSortEvents(0, total - 1);
  }

//+------------------------------------------------------------------+
bool LoadNews()
  {
   int handle = OpenCsv(InpNewsCsvFileName, FILE_READ | FILE_CSV | FILE_ANSI);
   if(handle == INVALID_HANDLE)
     {
      Print("Failed to open news CSV: ", InpNewsCsvFileName, " err=", GetLastError());
      return false;
     }

   string first_header = "";
   if(!FileIsEnding(handle))
      first_header = CleanText(FileReadString(handle));
   bool has_id = (first_header == "Id" || first_header == "ID" || first_header == "id");
   while(!FileIsLineEnding(handle) && !FileIsEnding(handle))
      FileReadString(handle);

   int count = 0;
   while(!FileIsEnding(handle))
     {
      string first_value = FileReadString(handle);
      if(StringLen(first_value) == 0 && FileIsEnding(handle))
         break;

      string datetime_value = (has_id ? FileReadString(handle) : first_value);
      string time_value = FileReadString(handle);
      string name_value = FileReadString(handle);
      string country_value = FileReadString(handle);
      string country_name = FileReadString(handle);
      string importance_value = FileReadString(handle);
      string actual_value = FileReadString(handle);
      string forecast_value = FileReadString(handle);
      string previous_value = FileReadString(handle);
      string impact_value = FileReadString(handle);
      string url_value = FileReadString(handle);

      datetime event_time = ParseEventTime(datetime_value, time_value);
      if(event_time > 0)
        {
         double actual = 0.0, forecast = 0.0, previous = 0.0;
         ArrayResize(gEventTimes, count + 1);
         ArrayResize(gEventNames, count + 1);
         ArrayResize(gEventCountries, count + 1);
         ArrayResize(gEventImportances, count + 1);
         ArrayResize(gEventImpacts, count + 1);
         ArrayResize(gActuals, count + 1);
         ArrayResize(gForecasts, count + 1);
         ArrayResize(gPrevious, count + 1);
         ArrayResize(gHasActuals, count + 1);
         ArrayResize(gHasForecasts, count + 1);
         ArrayResize(gHasPrevious, count + 1);

         gEventTimes[count] = event_time;
         gEventNames[count] = CsvText(name_value);
         gEventCountries[count] = UpperText(country_value);
         gEventImportances[count] = (int)StringToInteger(importance_value);
         gEventImpacts[count] = (int)StringToInteger(impact_value);
         gHasActuals[count] = ParseNumber(actual_value, actual);
         gHasForecasts[count] = ParseNumber(forecast_value, forecast);
         gHasPrevious[count] = ParseNumber(previous_value, previous);
         gActuals[count] = actual;
         gForecasts[count] = forecast;
         gPrevious[count] = previous;
         count++;
        }

      while(!FileIsLineEnding(handle) && !FileIsEnding(handle))
         FileReadString(handle);
     }

   FileClose(handle);
   if(count > 1)
      SortLoadedNewsEvents();
   gJournalScanIndex = 0;
   gTradeScanIndex = 0;
   Print("NewsEventsModelTester loaded ", count, " news rows from ", InpNewsCsvFileName);
   return count > 0;
  }

//+------------------------------------------------------------------+
bool LoadCsvSignals()
  {
   if(gUseOnnxRuntime)
      return true;

   int handle = OpenCsv(gEffectiveCsvSignalFile, FILE_READ | FILE_CSV | FILE_ANSI);
   if(handle == INVALID_HANDLE)
     {
      Print("Failed to open CSV signal table: ", gEffectiveCsvSignalFile, " err=", GetLastError());
      return false;
     }

   while(!FileIsLineEnding(handle) && !FileIsEnding(handle))
      FileReadString(handle);

   int count = 0;
   int scanned = 0;
   string active_symbol = UpperText(_Symbol);
   string active_model = UpperText(gEffectiveCsvModelName);
   while(!FileIsEnding(handle))
     {
      string event_time = FileReadString(handle);
      if(StringLen(event_time) == 0 && FileIsEnding(handle))
         break;
      string symbol = FileReadString(handle);
      string country = FileReadString(handle);
      string event_name = FileReadString(handle);
      string probability_text = FileReadString(handle);
      string signal_text = FileReadString(handle);
      string model_name = FileReadString(handle);
      scanned++;

      if(UpperText(symbol) != active_symbol ||
         (StringLen(active_model) > 0 && UpperText(model_name) != active_model))
        {
         while(!FileIsLineEnding(handle) && !FileIsEnding(handle))
            FileReadString(handle);
         continue;
        }

      ArrayResize(gCsvKeys, count + 1);
      ArrayResize(gCsvSymbols, count + 1);
      ArrayResize(gCsvProbabilities, count + 1);
      ArrayResize(gCsvSignals, count + 1);
      ArrayResize(gCsvModels, count + 1);
      gCsvKeys[count] = CleanText(event_time) + "|" + UpperText(country) + "|" + CsvText(event_name) + "|" + UpperText(symbol);
      gCsvSymbols[count] = UpperText(symbol);
      gCsvProbabilities[count] = StringToDouble(probability_text);
      gCsvSignals[count] = (int)StringToInteger(signal_text);
      gCsvModels[count] = CsvText(model_name);
      count++;

      while(!FileIsLineEnding(handle) && !FileIsEnding(handle))
         FileReadString(handle);
     }

   FileClose(handle);
   Print("NewsEventsModelTester loaded ", count, " CSV signal rows from ", gEffectiveCsvSignalFile,
         " | scanned=", scanned,
         " | symbol_filter=", active_symbol,
         " | model_filter=", gEffectiveCsvModelName);
   if(count == 0)
      Print("NewsEventsModelTester warning: no CSV signals for active symbol/model. Tester will continue without trades for this symbol.");
   return true;
  }

//+------------------------------------------------------------------+
bool InitOnnx()
  {
   if(!gUseOnnxRuntime)
      return true;

   uint flags = ONNX_COMMON_FOLDER | ONNX_USE_CPU_ONLY;
   if(!InpUseCommonFolder)
      flags = ONNX_USE_CPU_ONLY;

   gOnnxModel = OnnxCreate(gEffectiveOnnxModelFile, flags);
   if(gOnnxModel == INVALID_HANDLE)
     {
      Print("ONNX load failed: ", gEffectiveOnnxModelFile, " err=", GetLastError());
      return false;
     }

   const long input_shape[] = {1, 42};
   if(!OnnxSetInputShape(gOnnxModel, 0, input_shape))
     {
      Print("ONNX input shape failed err=", GetLastError());
      return false;
     }

   const long label_shape[] = {1};
   const long proba_shape[] = {1, 3};
   const long proba_shape_binary[] = {1, 2};
   OnnxSetOutputShape(gOnnxModel, 0, label_shape);
   if(InpModelSelection == NEWS_MODEL_V2_CATBOOST_PAIR_ONNX && !IsCatBoostPairMulticlass())
      OnnxSetOutputShape(gOnnxModel, 1, proba_shape_binary);
   else
      OnnxSetOutputShape(gOnnxModel, 1, proba_shape);
   Print("NewsEventsModelTester loaded ONNX model: ", gEffectiveOnnxModelFile);
   return true;
  }

//+------------------------------------------------------------------+
int CategoryIndex(const string event_name)
  {
   string n = UpperText(event_name);
   if(StringFind(n, "CPI") >= 0 || StringFind(n, "INFLATION") >= 0 || StringFind(n, "PCE") >= 0 || StringFind(n, "PPI") >= 0) return 31;
   if(StringFind(n, "EMPLOYMENT") >= 0 || StringFind(n, "PAYROLL") >= 0 || StringFind(n, "UNEMPLOYMENT") >= 0 || StringFind(n, "JOBLESS") >= 0 || StringFind(n, "CLAIMANT") >= 0 || StringFind(n, "WAGE") >= 0) return 32;
   if(StringFind(n, "GDP") >= 0 || StringFind(n, "PRODUCTION") >= 0 || StringFind(n, "RETAIL SALES") >= 0 || StringFind(n, "INDUSTRIAL") >= 0) return 33;
   if(StringFind(n, "INTEREST RATE") >= 0 || StringFind(n, "RATE DECISION") >= 0 || StringFind(n, "FOMC") >= 0 || StringFind(n, "ECB") >= 0 || StringFind(n, "BOE") >= 0 || StringFind(n, "BOJ") >= 0 || StringFind(n, "CENTRAL BANK") >= 0) return 34;
   if(StringFind(n, "PMI") >= 0 || StringFind(n, "CONFIDENCE") >= 0 || StringFind(n, "SENTIMENT") >= 0) return 35;
   if(StringFind(n, "TRADE BALANCE") >= 0 || StringFind(n, "EXPORTS") >= 0 || StringFind(n, "IMPORTS") >= 0) return 36;
   if(StringFind(n, "HOUSING") >= 0 || StringFind(n, "HOME SALE") >= 0) return 37;
   if(StringFind(n, "INVENTORIES") >= 0 || StringFind(n, "OIL") >= 0 || StringFind(n, "CRUDE") >= 0) return 38;
   return 39;
  }

//+------------------------------------------------------------------+
void BuildFeatures(const int index, float &features[])
  {
   ArrayInitialize(features, 0.0f);

   double surprise_fc = (gHasActuals[index] && gHasForecasts[index] ? gActuals[index] - gForecasts[index] : 0.0);
   double surprise_prev = (gHasActuals[index] && gHasPrevious[index] ? gActuals[index] - gPrevious[index] : 0.0);
   double pct_fc = (gHasActuals[index] && gHasForecasts[index] && MathAbs(gForecasts[index]) > 0.0000001 ? surprise_fc / MathAbs(gForecasts[index]) : 0.0);
   MqlDateTime dt;
   TimeToStruct(gEventTimes[index], dt);

   features[0] = (float)surprise_fc;
   features[1] = (float)surprise_prev;
   features[2] = (float)pct_fc;
   features[3] = (float)gEventImportances[index];
   features[4] = (float)gEventImpacts[index];
   features[5] = (float)dt.hour;

   string symbol = UpperText(_Symbol);
   string pairs[9] = {"EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","NZDUSD","USDCHF","EURJPY","GBPJPY"};
   for(int i = 0; i < 9; i++)
      if(symbol == pairs[i])
         features[6 + i] = 1.0f;

   string countries[8] = {"US","EU","GB","JP","AU","CA","NZ","CH"};
   for(int i = 0; i < 8; i++)
     {
      if(gEventCountries[index] == countries[i])
        {
         features[15 + i] = 1.0f;
         features[23 + i] = 1.0f;
        }
     }

   int cat = CategoryIndex(gEventNames[index]);
   features[cat] = 1.0f;

   string event_currency = CountryToCurrency(gEventCountries[index]);
   if(event_currency == gBaseCurrency)
      features[40] = 1.0f;
   if(event_currency == gQuoteCurrency)
      features[41] = 1.0f;
  }

//+------------------------------------------------------------------+
bool PredictOnnxDetailed(const int index,
                         double &probability,
                         int &domestic_signal,
                         double &prob_sell,
                         double &prob_no_trade,
                         double &prob_buy,
                         string &reason)
  {
   float x[42];
   BuildFeatures(index, x);

   long label[1] = {0};
   double buy_threshold = INTERNAL_PROBABILITY_BUY;
   double sell_threshold = INTERNAL_PROBABILITY_SELL;

   if(InpModelSelection == NEWS_MODEL_V2_CATBOOST_PAIR_ONNX && !IsCatBoostPairMulticlass())
     {
      buy_threshold = CatBoostPairThreshold();
      sell_threshold = CatBoostPairThreshold();
      float binary_probabilities[2] = {0.0f, 0.0f};
      if(!OnnxRun(gOnnxModel, ONNX_DEFAULT, x, label, binary_probabilities))
        {
         Print("ONNX run failed err=", GetLastError());
         reason = "PREDICTION_FAILED";
         return false;
        }

      prob_sell = (double)binary_probabilities[0];
      prob_no_trade = 0.0;
      prob_buy = (double)binary_probabilities[1];
      probability = MathMax(prob_sell, prob_buy);
      if(prob_buy >= prob_sell && prob_buy >= buy_threshold)
        {
         domestic_signal = 1;
         reason = "BUY";
        }
      else if(prob_sell > prob_buy && prob_sell >= sell_threshold)
        {
         domestic_signal = -1;
         reason = "SELL";
        }
      else
        {
         domestic_signal = 0;
         reason = "LOW_CONFIDENCE";
        }
      return true;
     }

   float probabilities[3] = {0.0f, 0.0f, 0.0f};
   if(!OnnxRun(gOnnxModel, ONNX_DEFAULT, x, label, probabilities))
     {
      Print("ONNX run failed err=", GetLastError());
      reason = "PREDICTION_FAILED";
      return false;
     }

   prob_sell = (double)probabilities[0];
   prob_no_trade = (double)probabilities[1];
   prob_buy = (double)probabilities[2];

   probability = MathMax(prob_sell, MathMax(prob_no_trade, prob_buy));

   bool buy_override = (INTERNAL_ALLOW_NO_TRADE_OVERRIDE &&
                        prob_buy >= INTERNAL_DIRECTIONAL_OVERRIDE_PROBABILITY &&
                        prob_buy >= prob_sell &&
                        (prob_no_trade - prob_buy) <= INTERNAL_NO_TRADE_DOMINANCE_MARGIN);
   bool sell_override = (INTERNAL_ALLOW_NO_TRADE_OVERRIDE &&
                         prob_sell >= INTERNAL_DIRECTIONAL_OVERRIDE_PROBABILITY &&
                         prob_sell >= prob_buy &&
                         (prob_no_trade - prob_sell) <= INTERNAL_NO_TRADE_DOMINANCE_MARGIN);

   if(prob_buy >= prob_sell && (prob_buy >= prob_no_trade || buy_override))
     {
      if(prob_buy >= buy_threshold)
        {
         domestic_signal = 1;
         reason = (prob_buy >= prob_no_trade ? "BUY" : "BUY_NO_TRADE_OVERRIDE");
        }
      else
        {
         domestic_signal = 0;
         reason = "LOW_CONFIDENCE";
        }
     }
   else if(prob_sell >= prob_buy && (prob_sell >= prob_no_trade || sell_override))
     {
      if(prob_sell >= sell_threshold)
        {
         domestic_signal = -1;
         reason = (prob_sell >= prob_no_trade ? "SELL" : "SELL_NO_TRADE_OVERRIDE");
        }
      else
        {
         domestic_signal = 0;
         reason = "LOW_CONFIDENCE";
        }
     }
   else if(prob_no_trade >= prob_buy && prob_no_trade >= prob_sell)
     {
      domestic_signal = 0;
      reason = "NO_TRADE_CLASS";
     }
   else
     {
      domestic_signal = 0;
      reason = "LOW_CONFIDENCE";
     }
   return true;
  }

//+------------------------------------------------------------------+
bool PredictCsv(const int index, double &probability, int &domestic_signal)
  {
   string key = EventKey(index);
   string key_upper = UpperText(key);
   for(int i = 0; i < ArraySize(gCsvKeys); i++)
     {
      if(StringLen(gEffectiveCsvModelName) > 0 && UpperText(gCsvModels[i]) != UpperText(gEffectiveCsvModelName))
         continue;
      if(UpperText(gCsvKeys[i]) == key_upper)
        {
         probability = gCsvProbabilities[i];
         domestic_signal = gCsvSignals[i];
         return true;
        }
     }
   probability = 0.0;
   domestic_signal = 0;
   return false;
  }

//+------------------------------------------------------------------+
int ToTradeSignal(const int index, const int domestic_signal)
  {
   if(domestic_signal == 0)
      return 0;
   string event_currency = CountryToCurrency(gEventCountries[index]);
   if(event_currency == gBaseCurrency)
      return domestic_signal;
   if(event_currency == gQuoteCurrency)
      return -domestic_signal;
   return 0;
  }

//+------------------------------------------------------------------+
bool IsRelevantEvent(const int index)
  {
   string event_currency = CountryToCurrency(gEventCountries[index]);
   return (event_currency == gBaseCurrency || event_currency == gQuoteCurrency);
  }

string SignalText(const int signal)
  {
   if(signal > 0)
      return "BUY";
   if(signal < 0)
      return "SELL";
   return "NO_TRADE";
  }

//+------------------------------------------------------------------+
string FormatNumberState(const bool has_value, const double value)
  {
   if(!has_value)
      return "";
   return DoubleToString(value, 8);
  }

//+------------------------------------------------------------------+
void JournalRelevantEvents()
  {
   if(!InpJournalRelevantEvents)
      return;

   datetime candle_open = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(candle_open <= 0 || candle_open == gLastJournalBarTime)
      return;

   int period_seconds = PeriodSeconds((ENUM_TIMEFRAMES)_Period);
   if(period_seconds <= 0)
      period_seconds = 3600;

   datetime candle_close = candle_open + period_seconds;
   gLastJournalBarTime = candle_open;

   while(gJournalScanIndex < ArraySize(gEventTimes) && gEventTimes[gJournalScanIndex] < candle_open)
      gJournalScanIndex++;

   for(int i = gJournalScanIndex; i < ArraySize(gEventTimes); i++)
     {
      if(gEventTimes[i] >= candle_close)
         break;
      if(!IsRelevantEvent(i))
         continue;

      long minutes_from_candle_open = (long)((gEventTimes[i] - candle_open) / 60);

      if(InpJournalPredictionDetails)
        {
         double probability = 0.0, prob_sell = 0.0, prob_no_trade = 0.0, prob_buy = 0.0;
         int domestic_signal = 0;
         string reason = "";
         bool predicted = false;
         if(gUseOnnxRuntime)
            predicted = PredictOnnxDetailed(i, probability, domestic_signal, prob_sell, prob_no_trade, prob_buy, reason);
         else
           {
            predicted = PredictCsv(i, probability, domestic_signal);
            reason = (predicted ? SignalText(domestic_signal) : "PREDICTION_FAILED");
           }

         int trade_signal = (predicted ? ToTradeSignal(i, domestic_signal) : 0);
         int spread = (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);

         if(predicted && trade_signal == 0 && reason != "PREDICTION_FAILED" && reason != "LOW_CONFIDENCE" && reason != "NO_TRADE_CLASS")
            reason = "NO_TRADE_CLASS";

         Print("NEWS_EVENT_CANDLE",
               " | candle_open=", TimeToString(candle_open, TIME_DATE | TIME_MINUTES),
               " | candle_close=", TimeToString(candle_close, TIME_DATE | TIME_MINUTES),
               " | minutes_from_candle_open=", (string)minutes_from_candle_open,
               " | symbol=", _Symbol,
               " | countries=", gBaseCountry, "/", gQuoteCountry,
               " | event_time=", TimeToString(gEventTimes[i], TIME_DATE | TIME_MINUTES),
               " | country=", gEventCountries[i],
               " | event=", gEventNames[i],
               " | actual=", FormatNumberState(gHasActuals[i], gActuals[i]),
               " | forecast=", FormatNumberState(gHasForecasts[i], gForecasts[i]),
               " | previous=", FormatNumberState(gHasPrevious[i], gPrevious[i]),
               " | prob_sell=", DoubleToString(prob_sell, 6),
               " | prob_no_trade=", DoubleToString(prob_no_trade, 6),
               " | prob_buy=", DoubleToString(prob_buy, 6),
               " | selected_class=", SignalText(domestic_signal),
               " | domestic_signal=", (string)domestic_signal,
               " | trade_signal=", SignalText(trade_signal),
               " | spread=", (string)spread,
               " | reason=", reason);
         gLastMonitoredEvent = TimeToString(gEventTimes[i], TIME_DATE | TIME_MINUTES) + " | " +
                               gEventCountries[i] + " | " + gEventNames[i] + " | " + reason;
        }
      else
        {
         Print("NEWS_EVENT_CANDLE",
               " | candle_open=", TimeToString(candle_open, TIME_DATE | TIME_MINUTES),
               " | event_time=", TimeToString(gEventTimes[i], TIME_DATE | TIME_MINUTES),
               " | symbol=", _Symbol,
               " | country=", gEventCountries[i],
               " | event=", gEventNames[i],
               " | actual=", FormatNumberState(gHasActuals[i], gActuals[i]),
               " | forecast=", FormatNumberState(gHasForecasts[i], gForecasts[i]),
               " | previous=", FormatNumberState(gHasPrevious[i], gPrevious[i]));
         gLastMonitoredEvent = TimeToString(gEventTimes[i], TIME_DATE | TIME_MINUTES) + " | " +
                               gEventCountries[i] + " | " + gEventNames[i];
        }
     }
  }

//+------------------------------------------------------------------+
string NextRelevantEventText()
  {
   datetime now = TimeCurrent();
   while(gNextEventScanIndex < ArraySize(gEventTimes) && gEventTimes[gNextEventScanIndex] < now)
      gNextEventScanIndex++;

   for(int i = gNextEventScanIndex; i < ArraySize(gEventTimes); i++)
     {
      if(!IsRelevantEvent(i))
         continue;
      return TimeToString(gEventTimes[i], TIME_DATE | TIME_MINUTES) + " | " +
             gEventCountries[i] + " | " + gEventNames[i];
     }
   return "none";
  }

//+------------------------------------------------------------------+
string CommentForEvent(const int index)
  {
   string comment = gEventNames[index];
   if(StringLen(comment) > 28)
      comment = StringSubstr(comment, 0, 28);
   return comment;
  }

//+------------------------------------------------------------------+
bool AppendPrediction(const int index, const double probability, const int domestic_signal, const int trade_signal, const string status, const ulong order_ticket=0, const ulong position_id=0)
  {
   bool exists = (InpUseCommonFolder ? FileIsExist(gPredictionCsv, FILE_COMMON) : FileIsExist(gPredictionCsv));
   int handle = OpenCsv(gPredictionCsv, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI);
   if(handle == INVALID_HANDLE)
      return false;
   FileSeek(handle, 0, SEEK_END);
   if(!exists || FileTell(handle) == 0)
      FileWrite(handle, "SERVER_TIME", "EVENT_TIME", "SYMBOL", "COUNTRY", "EVENT_NAME", "MODEL_RUNTIME", "ACTUAL", "FORECAST", "PREVIOUS", "PROBABILITY", "DOMESTIC_SIGNAL", "TRADE_SIGNAL", "SPREAD", "ORDER_TICKET", "POSITION_ID", "STATUS");
   FileWrite(handle,
             TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
             TimeToString(gEventTimes[index], TIME_DATE | TIME_SECONDS),
             _Symbol,
             gEventCountries[index],
             CsvText(gEventNames[index]),
             ModelSelectionName(),
             DoubleToString(gActuals[index], 8),
             DoubleToString(gForecasts[index], 8),
             DoubleToString(gPrevious[index], 8),
             DoubleToString(probability, 8),
             IntegerToString(domestic_signal),
             IntegerToString(trade_signal),
             IntegerToString((int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD)),
             (string)order_ticket,
             (string)position_id,
             CsvText(status));
   FileClose(handle);
   return true;
  }

//+------------------------------------------------------------------+
int CountPositions(int &buy_count, int &sell_count)
  {
   buy_count = 0;
   sell_count = 0;
   int total = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      total++;
      ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      if(type == POSITION_TYPE_BUY)
         buy_count++;
      if(type == POSITION_TYPE_SELL)
         sell_count++;
     }
   return total;
  }

//+------------------------------------------------------------------+
void CloseOpposite(const int signal)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      if((signal > 0 && type == POSITION_TYPE_SELL) || (signal < 0 && type == POSITION_TYPE_BUY))
         trade.PositionClose(ticket);
     }
  }

//+------------------------------------------------------------------+
void CloseExpiredPositions()
  {
   if(InpExitMode == EXIT_SL_TP || InpHoldingHours <= 0)
      return;

   datetime now = TimeCurrent();
   int holding_seconds = InpHoldingHours * 3600;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;

      ulong position_id = (ulong)PositionGetInteger(POSITION_IDENTIFIER);
      datetime open_time = LookupPositionOpenTime(position_id);
      if(open_time <= 0)
         open_time = (datetime)PositionGetInteger(POSITION_TIME);
      if(open_time <= 0)
         continue;

      if(now - open_time >= holding_seconds)
        {
         if(trade.PositionClose(ticket))
           {
            gLastStatus = "TIME_EXIT | position=" + (string)position_id + " | held_hours=" + (string)InpHoldingHours;
            if(InpEnableLog)
               Print(gLastStatus);
           }
        }
     }
  }

//+------------------------------------------------------------------+
double NormalizeVolumeToSymbol(const double volume)
  {
   double min_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(step <= 0.0)
      step = 0.01;

   double normalized = MathFloor(volume / step) * step;
   if(normalized < min_volume)
      normalized = min_volume;
   if(normalized > max_volume)
      normalized = max_volume;

   int digits = 2;
   if(step < 0.01)
      digits = 3;
   if(step < 0.001)
      digits = 4;
   return NormalizeDouble(normalized, digits);
  }

//+------------------------------------------------------------------+
double CalculateRiskVolume(const double entry_price, const double sl_price)
  {
   if(InpRiskPercent <= 0.0 || InpStopLossPoints <= 0)
      return 0.0;

   double risk_money = AccountInfoDouble(ACCOUNT_EQUITY) * InpRiskPercent / 100.0;
   double profit_at_sl = 0.0;
   ENUM_ORDER_TYPE order_type = (sl_price < entry_price ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
   if(!OrderCalcProfit(order_type, _Symbol, 1.0, entry_price, sl_price, profit_at_sl))
     {
      Print("Risk lot calculation failed | symbol=", _Symbol, " | err=", GetLastError());
      return 0.0;
     }

   double loss_per_lot = MathAbs(profit_at_sl);
   if(loss_per_lot <= 0.0)
      return 0.0;

   return NormalizeVolumeToSymbol(risk_money / loss_per_lot);
  }

//+------------------------------------------------------------------+
double CalculateTradeVolume(const double entry_price, const double sl_price)
  {
   if(InpLotMode == LOT_MODE_FIXED)
      return NormalizeVolumeToSymbol(InpFixedLot);
   if(InpExitMode == EXIT_AFTER_HOURS)
     {
      Print("Dynamic risk requires SL. EXIT_AFTER_HOURS uses fixed lot fallback: ", DoubleToString(InpFixedLot, 2));
      return NormalizeVolumeToSymbol(InpFixedLot);
     }
   return CalculateRiskVolume(entry_price, sl_price);
  }

//+------------------------------------------------------------------+
bool ExecuteTrade(const int signal, const string comment, ulong &order_ticket, ulong &position_id)
  {
   order_ticket = 0;
   position_id = 0;
   if(signal == 0)
      return false;

   int buy_count = 0, sell_count = 0;
   int total = CountPositions(buy_count, sell_count);
   if(total > 0)
     {
      if(!InpAllowReverse)
         return false;
      if((signal > 0 && buy_count > 0) || (signal < 0 && sell_count > 0))
         return false;
      CloseOpposite(signal);
     }

   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl = 0.0, tp = 0.0;
   double volume = 0.0;
   trade.SetExpertMagicNumber(InpMagicNumber);

   bool ok = false;
   bool use_sl_tp = (InpExitMode == EXIT_SL_TP || InpExitMode == EXIT_SL_TP_OR_AFTER_HOURS);
   if(signal > 0)
     {
      sl = (use_sl_tp && InpStopLossPoints > 0 ? NormalizeDouble(ask - InpStopLossPoints * point, digits) : 0.0);
      tp = (use_sl_tp && InpTakeProfitPoints > 0 ? NormalizeDouble(ask + InpTakeProfitPoints * point, digits) : 0.0);
      volume = CalculateTradeVolume(ask, sl);
      if(volume <= 0.0)
        {
         Print("Trade skipped: invalid volume | lot_mode=", (InpLotMode == LOT_MODE_FIXED ? "fixed" : "dynamic_risk"),
               " | fixed_lot=", DoubleToString(InpFixedLot, 2),
               " | risk_percent=", DoubleToString(InpRiskPercent, 2),
               " | sl_points=", (string)InpStopLossPoints);
         return false;
        }
      ok = trade.Buy(volume, _Symbol, ask, sl, tp, comment);
     }
   else
     {
      sl = (use_sl_tp && InpStopLossPoints > 0 ? NormalizeDouble(bid + InpStopLossPoints * point, digits) : 0.0);
      tp = (use_sl_tp && InpTakeProfitPoints > 0 ? NormalizeDouble(bid - InpTakeProfitPoints * point, digits) : 0.0);
      volume = CalculateTradeVolume(bid, sl);
      if(volume <= 0.0)
        {
         Print("Trade skipped: invalid volume | lot_mode=", (InpLotMode == LOT_MODE_FIXED ? "fixed" : "dynamic_risk"),
               " | fixed_lot=", DoubleToString(InpFixedLot, 2),
               " | risk_percent=", DoubleToString(InpRiskPercent, 2),
               " | sl_points=", (string)InpStopLossPoints);
         return false;
        }
      ok = trade.Sell(volume, _Symbol, bid, sl, tp, comment);
     }

   order_ticket = trade.ResultOrder();
   if(trade.ResultDeal() > 0 && HistoryDealSelect(trade.ResultDeal()))
      position_id = (ulong)HistoryDealGetInteger(trade.ResultDeal(), DEAL_POSITION_ID);
   return ok;
  }

//+------------------------------------------------------------------+
bool AppendDeal(const ulong deal_ticket)
  {
   if(deal_ticket == 0 || !HistoryDealSelect(deal_ticket))
      return false;
   bool exists = (InpUseCommonFolder ? FileIsExist(gTransactionCsv, FILE_COMMON) : FileIsExist(gTransactionCsv));
   int handle = OpenCsv(gTransactionCsv, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI);
   if(handle == INVALID_HANDLE)
      return false;
   FileSeek(handle, 0, SEEK_END);
   if(!exists || FileTell(handle) == 0)
      FileWrite(handle, "TIME", "DEAL_TICKET", "ORDER_TICKET", "POSITION_ID", "SYMBOL", "DEAL_TYPE", "ENTRY", "VOLUME", "PRICE", "PROFIT", "SWAP", "COMMISSION", "COMMENT", "MODEL_RUNTIME", "EVENT_TIME", "EVENT_COUNTRY", "EVENT_NAME");

   ulong position_id = (ulong)HistoryDealGetInteger(deal_ticket, DEAL_POSITION_ID);
   datetime event_time = 0;
   string event_country = "";
   string event_name = "";
   string model_name = "";
   LookupPositionEvent(position_id, event_time, event_country, event_name, model_name);

   FileWrite(handle,
             TimeToString((datetime)HistoryDealGetInteger(deal_ticket, DEAL_TIME), TIME_DATE | TIME_SECONDS),
             (string)deal_ticket,
             (string)HistoryDealGetInteger(deal_ticket, DEAL_ORDER),
             (string)position_id,
             HistoryDealGetString(deal_ticket, DEAL_SYMBOL),
             (string)HistoryDealGetInteger(deal_ticket, DEAL_TYPE),
             (string)HistoryDealGetInteger(deal_ticket, DEAL_ENTRY),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_VOLUME), 2),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_PRICE), _Digits),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_PROFIT), 2),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_SWAP), 2),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION), 2),
             CsvText(HistoryDealGetString(deal_ticket, DEAL_COMMENT)),
             CsvText(model_name),
             (event_time > 0 ? TimeToString(event_time, TIME_DATE | TIME_SECONDS) : ""),
             CsvText(event_country),
             CsvText(event_name));
   FileClose(handle);
   return true;
  }

//+------------------------------------------------------------------+
void ExportHistory()
  {
   if(!HistorySelect(0, TimeCurrent()))
      return;
   int handle = OpenCsv(gTransactionCsv, FILE_WRITE | FILE_CSV | FILE_ANSI);
   if(handle == INVALID_HANDLE)
      return;
   FileWrite(handle, "TIME", "DEAL_TICKET", "ORDER_TICKET", "POSITION_ID", "SYMBOL", "DEAL_TYPE", "ENTRY", "VOLUME", "PRICE", "PROFIT", "SWAP", "COMMISSION", "COMMENT", "MODEL_RUNTIME", "EVENT_TIME", "EVENT_COUNTRY", "EVENT_NAME");
   int total = HistoryDealsTotal();
   for(int i = 0; i < total; i++)
     {
      ulong deal = HistoryDealGetTicket(i);
      if(deal == 0 || !HistoryDealSelect(deal))
         continue;
      if(HistoryDealGetString(deal, DEAL_SYMBOL) != _Symbol)
         continue;
      if((ulong)HistoryDealGetInteger(deal, DEAL_MAGIC) != InpMagicNumber)
         continue;
      ulong position_id = (ulong)HistoryDealGetInteger(deal, DEAL_POSITION_ID);
      datetime event_time = 0;
      string event_country = "";
      string event_name = "";
      string model_name = "";
      LookupPositionEvent(position_id, event_time, event_country, event_name, model_name);
      FileWrite(handle,
                TimeToString((datetime)HistoryDealGetInteger(deal, DEAL_TIME), TIME_DATE | TIME_SECONDS),
                (string)deal,
                (string)HistoryDealGetInteger(deal, DEAL_ORDER),
                (string)position_id,
                HistoryDealGetString(deal, DEAL_SYMBOL),
                (string)HistoryDealGetInteger(deal, DEAL_TYPE),
                (string)HistoryDealGetInteger(deal, DEAL_ENTRY),
                DoubleToString(HistoryDealGetDouble(deal, DEAL_VOLUME), 2),
                DoubleToString(HistoryDealGetDouble(deal, DEAL_PRICE), _Digits),
                DoubleToString(HistoryDealGetDouble(deal, DEAL_PROFIT), 2),
                DoubleToString(HistoryDealGetDouble(deal, DEAL_SWAP), 2),
                DoubleToString(HistoryDealGetDouble(deal, DEAL_COMMISSION), 2),
                CsvText(HistoryDealGetString(deal, DEAL_COMMENT)),
                CsvText(model_name),
                (event_time > 0 ? TimeToString(event_time, TIME_DATE | TIME_SECONDS) : ""),
                CsvText(event_country),
                CsvText(event_name));
     }
   FileClose(handle);
  }

//+------------------------------------------------------------------+
void ProcessEvents()
  {
   datetime now = TimeCurrent();

   while(gTradeScanIndex < ArraySize(gEventTimes) && now - gEventTimes[gTradeScanIndex] > InpTradeWindowMinutes * 60)
      gTradeScanIndex++;

   for(int i = gTradeScanIndex; i < ArraySize(gEventTimes); i++)
     {
      if(gEventTimes[i] > now)
         break;

      if(!IsRelevantEvent(i))
         continue;

      string key = EventKey(i);
      if(InpOneTradePerEvent && WasTraded(key))
         continue;

      if(!PassesModelEventFilter(i))
        {
         AppendPrediction(i, 0.0, 0, 0, "MODEL_EVENT_FILTER");
         MarkTraded(key);
         continue;
        }

      double probability = 0.0;
      int domestic_signal = 0;
      double prob_sell = 0.0, prob_no_trade = 0.0, prob_buy = 0.0;
      string reason = "";
      bool predicted = false;
      if(gUseOnnxRuntime)
         predicted = PredictOnnxDetailed(i, probability, domestic_signal, prob_sell, prob_no_trade, prob_buy, reason);
      else
        {
         predicted = PredictCsv(i, probability, domestic_signal);
         reason = (predicted ? SignalText(domestic_signal) : "PREDICTION_FAILED");
        }
      if(!predicted)
        {
         AppendPrediction(i, probability, domestic_signal, 0, "PREDICTION_FAILED");
         MarkTraded(key);
         continue;
        }

      int trade_signal = ToTradeSignal(i, domestic_signal);
      if(trade_signal == 0)
        {
         AppendPrediction(i, probability, domestic_signal, trade_signal, reason);
         MarkTraded(key);
         continue;
        }

      int spread = (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
      if(InpMaxSpreadPoints > 0 && spread > InpMaxSpreadPoints)
        {
         AppendPrediction(i, probability, domestic_signal, trade_signal, "SPREAD_TOO_HIGH");
         MarkTraded(key);
         continue;
        }

      ulong order_ticket = 0, position_id = 0;
      bool ok = ExecuteTrade(trade_signal, CommentForEvent(i), order_ticket, position_id);
      if(ok && position_id > 0)
         RegisterPositionEvent(position_id, i);
      string status = (ok ? "TRADE_EXECUTED" : "EXECUTION_FAILED_OR_SKIPPED");
      AppendPrediction(i, probability, domestic_signal, trade_signal, status, order_ticket, position_id);
      MarkTraded(key);
      gLastStatus = status + " | " + gEventNames[i] + " | signal=" + SignalText(trade_signal) + " | p=" + DoubleToString(probability, 4);
      if(InpEnableLog)
         Print(gLastStatus);
     }
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   gPredictionCsv = ResolvePredictionCsv();
   gTransactionCsv = ResolveTransactionCsv();
   gUseOnnxRuntime = IsOnnxSelection();
   gEffectiveOnnxModelFile = ResolveOnnxModelFile();
   gEffectiveCsvSignalFile = ResolveCsvSignalFile();
   gEffectiveCsvModelName = ResolveCsvModelName();
   gBaseCurrency = StringSubstr(_Symbol, 0, 3);
   gQuoteCurrency = StringSubstr(_Symbol, 3, 3);
   gBaseCountry = CurrencyToCountry(gBaseCurrency);
   gQuoteCountry = CurrencyToCountry(gQuoteCurrency);
   ArrayResize(gTradedKeys, 0);
   gLastJournalBarTime = 0;
   gJournalScanIndex = 0;
   gTradeScanIndex = 0;
   gNextEventScanIndex = 0;

   if(StringLen(gBaseCountry) == 0 || StringLen(gQuoteCountry) == 0)
     {
      Print("Unsupported FX symbol for NewsEventsModelTester: ", _Symbol);
      return INIT_FAILED;
     }

   if(!LoadNews())
      return INIT_FAILED;
   if(!LoadCsvSignals())
      return INIT_FAILED;
   if(!InitOnnx())
      return INIT_FAILED;

   Print("NewsEventsModelTester initialized | symbol=", _Symbol,
         " | model=", ModelSelectionName(),
         " | runtime=", (gUseOnnxRuntime ? "ONNX" : "CSV_SIGNAL_TABLE"),
         " | onnx_file=", gEffectiveOnnxModelFile,
         " | csv_file=", gEffectiveCsvSignalFile,
         " | csv_model=", gEffectiveCsvModelName,
         " | news=", InpNewsCsvFileName,
         " | prediction_csv=", gPredictionCsv,
         " | transaction_csv=", gTransactionCsv);
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(gOnnxModel != INVALID_HANDLE)
      OnnxRelease(gOnnxModel);
   if(InpExportHistoryOnDeinit)
      ExportHistory();
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   JournalRelevantEvents();
   CloseExpiredPositions();
   ProcessEvents();
   Comment("NewsEventsModelTester",
           "\nModel: ", ModelSelectionName(),
           "\nRuntime: ", (gUseOnnxRuntime ? "ONNX" : "CSV_SIGNAL_TABLE"),
           "\nExit mode: ", (InpExitMode == EXIT_AFTER_HOURS ? "EXIT_AFTER_HOURS" : (InpExitMode == EXIT_SL_TP_OR_AFTER_HOURS ? "SL_TP_OR_TIME" : "SL_TP")),
           "\nHolding hours: ", InpHoldingHours,
           "\nNews rows: ", ArraySize(gEventTimes),
           "\nRelevant countries: ", gBaseCountry, "/", gQuoteCountry,
           "\nNext relevant event: ", NextRelevantEventText(),
           "\nLast monitored: ", gLastMonitoredEvent,
           "\nLast: ", gLastStatus);
  }

//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
  {
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD && trans.deal > 0)
      AppendDeal(trans.deal);
  }
//+------------------------------------------------------------------+





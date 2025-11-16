import argparse
from pipelines.daily_forecast import run_daily_forecast

def main():
    parser = argparse.ArgumentParser(prog="eagle", description="Eagle-1 CLI")
    subparsers = parser.add_subparsers(dest="command")

    # ------- FORECAST COMMAND -------
    forecast_parser = subparsers.add_parser("forecast", help="Run forecast")
    forecast_parser.add_argument("--symbol", type=str, default="BTC", help="Crypto symbol")
    forecast_parser.add_argument("--days", type=int, default=30, help="Forecast days")

    args = parser.parse_args()

    if args.command == "forecast":
        run_daily_forecast(symbol=args.symbol, days=args.days)

if __name__ == "__main__":
    main()

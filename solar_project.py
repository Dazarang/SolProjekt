import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.interpolate import make_interp_spline


class ComprehensiveSolarAnalysis:
    def __init__(
        self, customer_columns_file, house_load_file, generation_file, ev_load_file
    ):
        """Initialize the analysis with file paths."""
        self.customer_ids = (
            pd.read_csv(customer_columns_file, header=None).iloc[0].values
        )
        self.house_load = pd.read_csv(house_load_file, header=None)
        self.generation = pd.read_csv(generation_file, header=None)
        self.ev_load = pd.read_csv(ev_load_file, header=None)

        # Create timestamp index
        self.create_timestamp_index()

    def create_timestamp_index(self):
        """Create timestamp index for the data (3 years with 30-min intervals)."""
        start_date = datetime(2019, 8, 2)  # Assuming start date based on pattern
        timestamps = [start_date + timedelta(minutes=30 * i) for i in range(52608)]

        # Set index for all dataframes
        self.house_load.index = timestamps
        self.generation.index = timestamps
        self.ev_load.index = timestamps

    def select_sample_houses(self, n_houses=5):
        """Select a sample of houses for analysis."""
        # np.random.seed(0)  # For reproducibility
        # self.selected_houses = np.random.choice(self.customer_ids, size=n_houses, replace=False)
        self.selected_houses = np.array([73, 82, 141])
        return self.selected_houses

    def calculate_metrics(self, load, generation, period_mask=None):
        """Calculate self-consumption and self-sufficiency using raw data."""
        if period_mask is not None:
            load = load[period_mask]
            generation = generation[period_mask]

        # Calculate overlap for each timestep (30-min intervals)
        overlap = np.minimum(load, generation)

        # Calculate metrics using raw data sums
        self_consumption = (np.sum(overlap) / np.sum(generation)) * 100
        self_sufficiency = (np.sum(overlap) / np.sum(load)) * 100

        return self_consumption, self_sufficiency

    def plot_daily_profile(self, house_id):
        """Create daily profile plot for a specific house with filled areas and smoothing."""
        # Get separate data for house and EV
        pure_house_load = self.house_load[house_id]
        ev_load = self.ev_load[house_id]
        total_load = pure_house_load + ev_load
        generation = self.generation[house_id]

        # Calculate average daily profiles
        daily_house_load = pure_house_load.groupby(pure_house_load.index.hour).mean()
        daily_ev_load = ev_load.groupby(ev_load.index.hour).mean()
        daily_total_load = total_load.groupby(total_load.index.hour).mean()
        daily_generation = generation.groupby(generation.index.hour).mean()

        # Create more granular x-axis for smoother interpolation
        x_smooth = np.linspace(0, 23, 200)

        # Interpolate the data for all components
        spl_house = make_interp_spline(
            daily_house_load.index, daily_house_load.values, k=3
        )
        spl_ev = make_interp_spline(daily_ev_load.index, daily_ev_load.values, k=3)
        spl_total = make_interp_spline(
            daily_total_load.index, daily_total_load.values, k=3
        )
        spl_gen = make_interp_spline(
            daily_generation.index, daily_generation.values, k=3
        )

        # Generate smooth curves
        y_house_smooth = np.maximum(spl_house(x_smooth), 0)
        y_ev_smooth = np.maximum(spl_ev(x_smooth), 0)
        y_total_smooth = np.maximum(spl_total(x_smooth), 0)
        y_gen_smooth = np.maximum(spl_gen(x_smooth), 0)

        # Calculate overlap (self-consumption)
        y_overlap_smooth = np.minimum(y_total_smooth, y_gen_smooth)

        plt.figure(figsize=(11.7, 8.3))

        # Plot house load
        plt.fill_between(
            x_smooth,
            0,
            y_house_smooth,
            alpha=0.4,
            color="#3182bd",
            label="House Load",
            edgecolor="#08519c",
            linewidth=1,
        )

        # Plot EV load stacked on top of house load
        plt.fill_between(
            x_smooth,
            y_house_smooth,
            y_house_smooth + y_ev_smooth,
            alpha=0.4,
            color="#756bb1",  # Purple for EV
            label="EV Load",
            edgecolor="#54278f",  # Darker purple edge
            linewidth=1,
        )

        # Plot generation
        plt.fill_between(
            x_smooth,
            0,
            y_gen_smooth,
            alpha=0.4,
            color="#fd8d3c",
            label="Generation",
            edgecolor="#d94801",
            linewidth=1,
        )

        # Plot overlap
        plt.fill_between(
            x_smooth,
            0,
            y_overlap_smooth,
            alpha=0.6,
            color="#31a354",
            label="Self-Consumption",
            edgecolor="#006d2c",
            linewidth=1,
        )

        plt.title(
            f"Average Daily Profile (3-Year Average) - House {house_id}", fontsize=16
        )
        plt.xlabel("Hour of Day", fontsize=14)
        plt.ylabel("Energy (kWh)", fontsize=14)
        plt.legend(fontsize=12, loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24), fontsize=12)
        plt.yticks(fontsize=12)

        # Calculate metrics using raw data
        self_consumption, self_sufficiency = self.calculate_metrics(
            total_load, generation
        )

        # Add text annotations for metrics
        plt.text(
            0.02,
            0.95,
            f"Self-Consumption: {self_consumption:.1f}%\n"
            + f"Self-Sufficiency: {self_sufficiency:.1f}%",
            transform=plt.gca().transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8),
            verticalalignment="top",
        )

        return plt.gcf()

    def plot_daily_average_for_season(self, house_id):
        """Create smoothed average daily profile plots for summer (Dec-Feb) and winter (Jun-Aug) in subplots."""
        # Get separate data for house and EV
        pure_house_load = self.house_load[house_id]
        ev_load = self.ev_load[house_id]
        total_load = pure_house_load + ev_load
        generation = self.generation[house_id]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.7, 16.5))

        # font sizes
        title_font = 16
        label_font = 14
        tick_font = 12
        legend_font = 12

        def process_seasonal_data(months, ax, season_name):
            seasonal_data = pd.DataFrame()
            year_count = 0

            # Create season mask for raw data metrics
            season_mask = pd.Series(False, index=total_load.index)

            # Process data for all available years
            years = [2019, 2020, 2021]  # All years in the dataset

            for year in years:
                for month in months:
                    # Adjust year for December (belongs to summer of next year)
                    year_to_use = year if month != 12 else year - 1

                    # Create mask for the specific month and year
                    mask = (pure_house_load.index.year == year_to_use) & (
                        pure_house_load.index.month == month
                    )
                    season_mask = season_mask | mask

                    # Skip if no data for this period
                    if not mask.any():
                        continue

                    # Group by hour and get mean for this month for each component (for visualization)
                    month_house = (
                        pure_house_load[mask]
                        .groupby(pure_house_load[mask].index.hour)
                        .mean()
                    )
                    month_ev = ev_load[mask].groupby(ev_load[mask].index.hour).mean()
                    month_total = (
                        total_load[mask].groupby(total_load[mask].index.hour).mean()
                    )
                    month_gen = (
                        generation[mask].groupby(generation[mask].index.hour).mean()
                    )

                    if seasonal_data.empty:
                        seasonal_data["house"] = month_house
                        seasonal_data["ev"] = month_ev
                        seasonal_data["total"] = month_total
                        seasonal_data["generation"] = month_gen
                    else:
                        seasonal_data["house"] += month_house
                        seasonal_data["ev"] += month_ev
                        seasonal_data["total"] += month_total
                        seasonal_data["generation"] += month_gen

                    year_count += 1

            # Calculate averages across all months and years (for visualization)
            seasonal_data = seasonal_data / year_count

            # Create more granular x-axis for smoother interpolation
            x_smooth = np.linspace(0, 23, 200)

            # Interpolate the data for smooth curves
            spl_house = make_interp_spline(
                seasonal_data.index, seasonal_data["house"].values, k=3
            )
            spl_ev = make_interp_spline(
                seasonal_data.index, seasonal_data["ev"].values, k=3
            )
            spl_total = make_interp_spline(
                seasonal_data.index, seasonal_data["total"].values, k=3
            )
            spl_gen = make_interp_spline(
                seasonal_data.index, seasonal_data["generation"].values, k=3
            )

            # Generate smooth curves
            y_house_smooth = np.maximum(spl_house(x_smooth), 0)
            y_ev_smooth = np.maximum(spl_ev(x_smooth), 0)
            y_total_smooth = np.maximum(spl_total(x_smooth), 0)
            y_gen_smooth = np.maximum(spl_gen(x_smooth), 0)

            # Calculate overlap for visualization
            y_overlap_smooth = np.minimum(y_total_smooth, y_gen_smooth)

            # Plot visualizations
            ax.fill_between(
                x_smooth,
                0,
                y_house_smooth,
                alpha=0.4,
                color="#3182bd",
                label="House Load",
                edgecolor="#08519c",
                linewidth=1,
            )

            ax.fill_between(
                x_smooth,
                y_house_smooth,
                y_house_smooth + y_ev_smooth,
                alpha=0.4,
                color="#756bb1",
                label="EV Load",
                edgecolor="#54278f",
                linewidth=1,
            )

            ax.fill_between(
                x_smooth,
                0,
                y_gen_smooth,
                alpha=0.4,
                color="#fd8d3c",
                label="Generation",
                edgecolor="#d94801",
                linewidth=1,
            )

            ax.fill_between(
                x_smooth,
                0,
                y_overlap_smooth,
                alpha=0.6,
                color="#31a354",
                label="Self-Consumption",
                edgecolor="#006d2c",
                linewidth=1,
            )

            # Calculate metrics using raw data
            self_consumption, self_sufficiency = self.calculate_metrics(
                total_load, generation, season_mask
            )

            ax.set_title(
                f"{season_name} Average Daily Profile (3-Year Average) - House {house_id}",
                fontsize=title_font,
            )
            ax.set_xlabel("Hour of Day", fontsize=label_font)
            ax.set_ylabel("Average Energy (kWh)", fontsize=label_font)
            ax.legend(fontsize=legend_font, loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(0, 24))
            ax.tick_params(axis="both", which="major", labelsize=tick_font)
            ax.set_xlim(0, 23)

            # Add self-consumption and self-sufficiency percentages
            ax.text(
                0.02,
                0.95,
                f"Seasonal Self-Consumption: {self_consumption:.1f}%\n"
                + f"Seasonal Self-Sufficiency: {self_sufficiency:.1f}%",
                transform=ax.transAxes,
                fontsize=tick_font,
                bbox=dict(facecolor="white", alpha=0.8),
                verticalalignment="top",
            )

            return self_consumption

        # Process Summer data (Dec-Feb)
        process_seasonal_data([12, 1, 2], ax1, "Summer (Dec-Feb)")

        # Process Winter data (Jun-Aug)
        process_seasonal_data([6, 7, 8], ax2, "Winter (Jun-Aug)")

        plt.tight_layout()
        return fig

    def plot_seasonal_data(self, house_id):
        """Plot seasonal averages across all years with filled areas and smoothing."""
        house_load = self.house_load[house_id] + self.ev_load[house_id]
        generation = self.generation[house_id]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.7, 16.5))

        # font sizes
        title_font = 16
        label_font = 14
        tick_font = 12
        legend_font = 12

        def process_seasonal_data(season_months, ax, season_name):
            seasonal_data = pd.DataFrame()
            year_count = 0

            # Create season mask for raw data metrics
            season_mask = pd.Series(False, index=house_load.index)

            # Process data for all available years
            years = [2019, 2020, 2021]  # All years in the dataset

            for year in years:
                for month in season_months:
                    # Adjust year for December (belongs to summer of next year)
                    year_to_use = year if month != 12 else year - 1

                    # Create mask for the specific month and year
                    mask = (house_load.index.year == year_to_use) & (
                        house_load.index.month == month
                    )
                    season_mask = season_mask | mask

                    # Skip if no data for this period
                    if not mask.any():
                        continue

                    month_load = house_load[mask].resample("h").sum()
                    month_gen = generation[mask].resample("h").sum()

                    if seasonal_data.empty:
                        seasonal_data["load"] = month_load
                        seasonal_data["generation"] = month_gen
                    else:
                        seasonal_data["load"] = seasonal_data["load"].add(
                            month_load, fill_value=0
                        )
                        seasonal_data["generation"] = seasonal_data["generation"].add(
                            month_gen, fill_value=0
                        )

                    year_count += 1

            # Calculate average across all months and years (for visualization)
            seasonal_data = seasonal_data / year_count
            days = np.arange(len(seasonal_data))

            # Create smoother x-axis
            x_smooth = np.linspace(0, len(days) - 1, 200)

            # Interpolate the data
            spl_load = make_interp_spline(days, seasonal_data["load"].values, k=3)
            spl_gen = make_interp_spline(days, seasonal_data["generation"].values, k=3)

            # Generate smooth curves
            y_load_smooth = spl_load(x_smooth)
            y_gen_smooth = spl_gen(x_smooth)

            # Ensure no negative values
            y_load_smooth = np.maximum(y_load_smooth, 0)
            y_gen_smooth = np.maximum(y_gen_smooth, 0)

            # Calculate overlap for visualization
            y_overlap_smooth = np.minimum(y_load_smooth, y_gen_smooth)

            # Plot smoothed data with improved colors and styling
            ax.fill_between(
                x_smooth,
                0,
                y_load_smooth,
                alpha=0.4,
                color="#3182bd",
                label="Daily consumption (load + EV)",
                edgecolor="#08519c",
                linewidth=1,
            )

            ax.fill_between(
                x_smooth,
                0,
                y_gen_smooth,
                alpha=0.4,
                color="#fd8d3c",
                label="Daily generation",
                edgecolor="#d94801",
                linewidth=1,
            )

            ax.fill_between(
                x_smooth,
                0,
                y_overlap_smooth,
                alpha=0.6,
                color="#31a354",
                label="Self-Consumption",
                edgecolor="#006d2c",
                linewidth=1,
            )

            # Calculate metrics using raw data
            self_consumption, self_sufficiency = self.calculate_metrics(
                house_load, generation, season_mask
            )

            ax.set_title(
                f"{season_name} - Average Hourly Energy (3-Year Average) - House {house_id}",
                fontsize=title_font,
            )
            ax.set_xlabel("Months in Season", fontsize=label_font)
            ax.set_ylabel("Average Energy (kWh)", fontsize=label_font)
            ax.legend(fontsize=legend_font, loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", which="major", labelsize=tick_font)

            # Add self-consumption and self-sufficiency percentages
            ax.text(
                0.02,
                0.95,
                f"Seasonal Self-Consumption: {self_consumption:.1f}%\n"
                + f"Seasonal Self-Sufficiency: {self_sufficiency:.1f}%",
                transform=ax.transAxes,
                fontsize=tick_font,
                bbox=dict(facecolor="white", alpha=0.8),
                verticalalignment="top",
            )

            # Set x-axis limits to match original data
            ax.set_xlim(0, len(days) - 1)

            # Set custom x-axis ticks and labels
            if season_name.startswith("Summer"):
                ax.set_xticks([0, len(days) - 1])
                ax.set_xticklabels(["Dec", "Feb"])
            elif season_name.startswith("Winter"):
                ax.set_xticks([0, len(days) - 1])
                ax.set_xticklabels(["Jun", "Aug"])

        # Process Summer data (Dec-Feb)
        process_seasonal_data([12, 1, 2], ax1, "Summer (Dec-Feb)")

        # Process Winter data (Jun-Aug)
        process_seasonal_data([6, 7, 8], ax2, "Winter (Jun-Aug)")

        plt.tight_layout()
        return fig

    def create_summary_report(self, selected_houses=None):
        """Create a summary report for selected houses showing metrics based on raw data."""
        if selected_houses is None:
            selected_houses = self.selected_houses

        metrics_list = []
        for house_id in selected_houses:
            # Get separate data components for house
            pure_house_load = self.house_load[house_id]
            ev_load = self.ev_load[house_id]
            total_load = pure_house_load + ev_load
            generation = self.generation[house_id]

            # Calculate metrics using raw data
            self_consumption, self_sufficiency = self.calculate_metrics(
                total_load, generation
            )

            # Calculate overlap for total self-consumed calculation
            overlap = np.minimum(total_load, generation)

            metrics = {
                "house_id": house_id,
                "self_consumption [%]": self_consumption,
                "self_sufficiency [%]": self_sufficiency,
                "total_house_load [kWh]": np.sum(pure_house_load) / 3,  # 3 years
                "total_ev_load [kWh]": np.sum(ev_load) / 3,
                "total_combined_load [kWh]": np.sum(total_load) / 3,
                "total_generation [kWh]": np.sum(generation) / 3,
                "total_self_consumed [kWh]": np.sum(overlap)
                / 3,  # Using raw data overlap
                "ev_percentage [%]": (np.sum(ev_load) / np.sum(total_load)) * 100,
            }
            metrics_list.append(metrics)

        return pd.DataFrame(metrics_list)

    def print_seasonal_stats(self, house_id):
        """Print seasonal statistics using raw data for metrics calculations."""
        pure_house_load = self.house_load[house_id]
        ev_load = self.ev_load[house_id]
        total_load = pure_house_load + ev_load
        generation = self.generation[house_id]

        def calculate_seasonal_stats(months):
            # Create season mask for raw data metrics
            season_mask = pd.Series(False, index=total_load.index)
            seasonal_data = pd.DataFrame()

            # Process data for all available years
            years = [2019, 2020, 2021]  # All years in the dataset

            for year in years:
                for month in months:
                    # Adjust year for December (belongs to summer of next year)
                    year_to_use = year if month != 12 else year - 1
                    mask = (pure_house_load.index.year == year_to_use) & (
                        pure_house_load.index.month == month
                    )
                    season_mask = season_mask | mask

                    # Skip if no data for this period
                    if not mask.any():
                        continue

                    # Get data for each component (for regular stats)
                    month_data = pd.DataFrame(
                        {
                            "house_load": pure_house_load[mask],
                            "ev_load": ev_load[mask],
                            "total_load": total_load[mask],
                            "generation": generation[mask],
                        }
                    )

                    if seasonal_data.empty:
                        seasonal_data = month_data
                    else:
                        seasonal_data = pd.concat([seasonal_data, month_data])

            # Calculate metrics using raw data
            self_consumption, self_sufficiency = self.calculate_metrics(
                total_load, generation, season_mask
            )

            # Calculate daily totals for daily statistics
            daily_data = seasonal_data.resample("D").sum()

            # Calculate EV percentage
            ev_percentage = (
                seasonal_data["ev_load"].sum() / seasonal_data["total_load"].sum()
            ) * 100

            return {
                "Total House Load (kWh)": seasonal_data["house_load"].sum() / 3,
                "Total EV Load (kWh)": seasonal_data["ev_load"].sum() / 3,
                "Total Combined Load (kWh)": seasonal_data["total_load"].sum() / 3,
                "Total Generation (kWh)": seasonal_data["generation"].sum() / 3,
                "Self-Consumption (%)": self_consumption,
                "Self-Sufficiency (%)": self_sufficiency,
                "EV Load Percentage (%)": ev_percentage,
                "Days with Excess Generation": sum(
                    daily_data["generation"] > daily_data["total_load"]
                )
                / 3,
            }

        # Calculate statistics for summer (Dec-Feb) and winter (Jun-Aug)
        stats = pd.DataFrame(
            {
                "Summer (Dec-Feb)": calculate_seasonal_stats([12, 1, 2]),
                "Winter (Jun-Aug)": calculate_seasonal_stats([6, 7, 8]),
            }
        )

        return stats


import os


def main():
    try:
        # Create directories
        os.makedirs("reports", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        os.makedirs("stats", exist_ok=True)

        # Initialize analysis
        analysis = ComprehensiveSolarAnalysis(
            "CustomerColumns.csv", "HouseLoad.csv", "Generation.csv", "EVLoad.csv"
        )

        # Select sample houses
        selected_houses = analysis.select_sample_houses(5)
        print("\nSelected houses for analysis:", selected_houses)

        # Generate and display 3 year summary report
        print("\nGenerating 3 year Summary Report...")
        summary_df = analysis.create_summary_report()
        print("\n3 year Summary Report:")
        print(summary_df.round(2))

        # Save 3 year summary report
        summary_df.to_csv("reports/3_year_summary_report.csv", index=False)
        print("\n3 year summary report saved to 'reports/3_year_summary_report.csv'")

        # Process each house
        print("\nDetailed analysis for each house...")
        for house_id in selected_houses:
            try:
                # Create and save daily profile plots
                daily_fig = analysis.plot_daily_profile(house_id)
                plt.savefig(
                    f"plots/house_{house_id}_daily_profile.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                # Create and save seasonal average plots
                seasonal_avg_fig = analysis.plot_daily_average_for_season(house_id)
                plt.savefig(
                    f"plots/house_{house_id}_seasonal_daily_averages.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                # Create and save seasonal profile plots
                seasonal_fig = analysis.plot_seasonal_data(house_id)
                plt.savefig(
                    f"plots/house_{house_id}_seasonal_profiles.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                # Generate and save seasonal statistics
                seasonal_stats = analysis.print_seasonal_stats(house_id)

                print(f"\nSeasonal Statistics for House {house_id}:")
                print(seasonal_stats.round(2))

                # Save seasonal statistics to CSV
                stats_filename = f"stats/house_{house_id}_seasonal_stats.csv"
                seasonal_stats.to_csv(stats_filename)
                print(f"Seasonal statistics saved to '{stats_filename}'")

                print(f"Successfully completed analysis for House {house_id}")

            except Exception as e:
                print(f"Error processing house {house_id}: {str(e)}")
                print("Continuing with next house...")
                continue

        print("\nAnalysis complete! All files have been saved.")

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")


if __name__ == "__main__":
    main()

"""Commands for analyzing pattern detection results."""

import typer
from typing import Optional
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from wave.cli import app

analyze_app = typer.Typer(name="analyze")
app.add_typer(analyze_app, name="analyze", help="Analyze pattern detection results")


@analyze_app.command("performance")
def analyze_performance(
    results_file: Path = typer.Argument(
        ..., help="JSON file containing detection results"
    ),
    ground_truth_file: Optional[Path] = typer.Option(
        None, "--truth", "-t", help="CSV or JSON file with ground truth labels"
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Path to save analysis results"
    ),
    visualize: bool = typer.Option(
        True, "--visualize/--no-visualize", help="Visualize performance metrics"
    ),
) -> None:
    """Analyze pattern detection performance against ground truth."""
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

    try:
        # Load detection results
        typer.echo(f"Loading detection results from {results_file}...")
        with open(results_file, 'r') as f:
            results = json.load(f)

        # Check if results is a list or a single result
        if not isinstance(results, list):
            results = [results]

        # Extract predictions
        predictions = []
        for result in results:
            if 'pattern_type' in result:
                predictions.append(result['pattern_type'])
            elif 'pattern_id' in result:
                # Try to convert pattern_id to numeric
                try:
                    if result['pattern_id'].startswith('ml_'):
                        pattern_id = result['pattern_id'][3:]
                        predictions.append(int(pattern_id) if pattern_id.isdigit() else pattern_id)
                    else:
                        predictions.append(result['pattern_id'])
                except:
                    predictions.append(result['pattern_id'])

        # Load ground truth if provided
        if ground_truth_file:
            typer.echo(f"Loading ground truth from {ground_truth_file}...")
            if ground_truth_file.suffix.lower() == '.csv':
                df = pd.read_csv(ground_truth_file)
                if 'label' in df.columns:
                    ground_truth = df['label'].values
                elif 'pattern_type' in df.columns:
                    ground_truth = df['pattern_type'].values
                else:
                    typer.echo(f"Error: Could not find label column in CSV file.")
                    raise typer.Exit(1)
            elif ground_truth_file.suffix.lower() == '.json':
                with open(ground_truth_file, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    if all(isinstance(item, dict) for item in data):
                        if 'label' in data[0]:
                            ground_truth = [item['label'] for item in data]
                        elif 'pattern_type' in data[0]:
                            ground_truth = [item['pattern_type'] for item in data]
                        else:
                            typer.echo(f"Error: Could not find label in JSON data.")
                            raise typer.Exit(1)
                    else:
                        ground_truth = data
                else:
                    typer.echo(f"Error: Unsupported JSON format.")
                    raise typer.Exit(1)

            # Validate data lengths
            if len(ground_truth) != len(predictions):
                typer.echo(f"Warning: Number of predictions ({len(predictions)}) doesn't match ground truth ({len(ground_truth)})")
                # Use minimum length
                min_len = min(len(ground_truth), len(predictions))
                ground_truth = ground_truth[:min_len]
                predictions = predictions[:min_len]

            # Calculate metrics
            typer.echo("Calculating performance metrics...")
            try:
                # For classification metrics, we need numeric or categorical labels
                accuracy = accuracy_score(ground_truth, predictions)
                report = classification_report(ground_truth, predictions, output_dict=True)
                cm = confusion_matrix(ground_truth, predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predictions, average='weighted')

                # Print metrics
                typer.echo(f"Performance metrics:")
                typer.echo(f"  Accuracy: {accuracy:.4f}")
                typer.echo(f"  Precision: {precision:.4f}")
                typer.echo(f"  Recall: {recall:.4f}")
                typer.echo(f"  F1 Score: {f1:.4f}")

                # Create result dictionary
                analysis_result = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "classification_report": report,
                    "confusion_matrix": cm.tolist()
                }

                # Save results if output file specified
                if output_file:
                    with open(output_file, 'w') as f:
                        json.dump(analysis_result, f, indent=2)
                    typer.echo(f"Analysis results saved to {output_file}")

                # Visualize if requested
                if visualize:
                    plt.figure(figsize=(10, 8))
                    plt.subplot(211)

                    # Plot confusion matrix
                    plt.imshow(cm, interpolation='nearest', cmap='Blues')
                    plt.title('Confusion Matrix')
                    plt.colorbar()

                    # Add labels
                    unique_labels = sorted(set(ground_truth).union(set(predictions)))
                    tick_marks = np.arange(len(unique_labels))
                    plt.xticks(tick_marks, unique_labels, rotation=45)
                    plt.yticks(tick_marks, unique_labels)

                    # Add values to the cells
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            plt.text(j, i, str(cm[i, j]),
                                    horizontalalignment="center",
                                    color="white" if cm[i, j] > cm.max() / 2 else "black")

                    plt.ylabel('True label')
                    plt.xlabel('Predicted label')

                    # Plot precision, recall, f1 scores
                    plt.subplot(212)
                    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                    values = [accuracy, precision, recall, f1]
                    bars = plt.bar(metrics, values)
                    plt.ylim(0, 1.0)
                    plt.ylabel('Score')
                    plt.title('Performance Metrics')

                    # Add values on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.4f}',
                                ha='center', va='bottom')

                    plt.tight_layout()
                    plt.show()

            except Exception as e:
                typer.echo(f"Error calculating metrics: {str(e)}")
                typer.echo("Falling back to simple comparison...")

                # Simple comparison
                correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
                total = len(predictions)
                accuracy = correct / total

                typer.echo(f"Simple accuracy: {accuracy:.4f} ({correct}/{total})")
        else:
            # Without ground truth, just summarize predictions
            typer.echo("No ground truth provided, summarizing predictions...")

            from collections import Counter
            prediction_counts = Counter(predictions)

            typer.echo(f"Prediction summary:")
            for pattern_type, count in prediction_counts.most_common():
                percentage = count / len(predictions) * 100
                typer.echo(f"  {pattern_type}: {count} ({percentage:.1f}%)")

            # Visualize if requested
            if visualize:
                plt.figure(figsize=(10, 6))

                # Sort by count for better visualization
                labels, values = zip(*prediction_counts.most_common())
                indexes = np.arange(len(labels))

                plt.bar(indexes, values)
                plt.xticks(indexes, labels, rotation=45)
                plt.title('Prediction Distribution')
                plt.xlabel('Pattern Type')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.show()

    except Exception as e:
        typer.echo(f"Error during performance analysis: {str(e)}")
        import traceback
        typer.echo(traceback.format_exc())
        raise typer.Exit(1)


@analyze_app.command("visualize")
def visualize_patterns(
    data_file: Path = typer.Argument(
        ..., help="CSV or JSON file containing time series data"
    ),
    results_file: Path = typer.Argument(
        ..., help="JSON file containing detection results"
    ),
    column: str = typer.Option(
        "close", "--column", "-c", help="Column name containing price data (for CSV)"
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Path to save visualization (PNG, PDF, SVG)"
    ),
) -> None:
    """Visualize detected patterns on time series data."""
    from wave.ml.viz.pattern_viz import PatternVisualizer

    try:
        # Load time series data
        typer.echo(f"Loading time series data from {data_file}...")
        if data_file.suffix.lower() == '.csv':
            df = pd.read_csv(data_file)
            if column not in df.columns:
                typer.echo(f"Error: Column '{column}' not found in CSV file.")
                raise typer.Exit(1)
            series = df[column].values
            dates = df['date'].values if 'date' in df.columns else None
        elif data_file.suffix.lower() == '.json':
            with open(data_file, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                series = np.array(data)
                dates = None
            elif isinstance(data, dict) and column in data:
                series = np.array(data[column])
                dates = np.array(data['date']) if 'date' in data else None
            else:
                typer.echo(f"Error: Could not find '{column}' data in JSON file.")
                raise typer.Exit(1)
        else:
            typer.echo(f"Error: Unsupported file format: {data_file.suffix}")
            raise typer.Exit(1)

        # Load detection results
        typer.echo(f"Loading detection results from {results_file}...")
        with open(results_file, 'r') as f:
            results = json.load(f)

        # Check if results is a list or a single result
        if not isinstance(results, list):
            results = [results]

        # Create pattern visualizer
        typer.echo("Creating visualization...")
        visualizer = PatternVisualizer()

        # Create figure
        plt.figure(figsize=(12, 8))

        # Plot time series
        if dates is not None:
            plt.plot(dates, series)
            plt.xlabel('Date')
        else:
            plt.plot(series)
            plt.xlabel('Time')

        plt.ylabel('Value')
        plt.title('Detected Patterns')

        # Add patterns to the plot
        for i, result in enumerate(results):
            pattern_type = result.get('pattern_type', result.get('pattern_id', f"Pattern {i+1}"))
            start_idx = result.get('start_idx', 0)
            end_idx = result.get('end_idx', len(series) - 1)
            confidence = result.get('confidence', result.get('score', 0.5))

            # Select color based on pattern type
            if isinstance(pattern_type, int):
                color = visualizer.get_color_for_pattern(pattern_type)
            else:
                color = visualizer.get_color_for_label(str(pattern_type))

            # Add pattern highlight
            visualizer.highlight_pattern(
                plt.gca(),
                x_values=range(start_idx, end_idx + 1) if dates is None else dates[start_idx:end_idx+1],
                y_values=series[start_idx:end_idx+1],
                pattern_label=str(pattern_type),
                confidence=confidence,
                color=color
            )

        plt.tight_layout()

        # Save if output file specified
        if output_file:
            plt.savefig(output_file)
            typer.echo(f"Visualization saved to {output_file}")

        # Show plot
        plt.show()

    except Exception as e:
        typer.echo(f"Error during visualization: {str(e)}")
        import traceback
        typer.echo(traceback.format_exc())
        raise typer.Exit(1)

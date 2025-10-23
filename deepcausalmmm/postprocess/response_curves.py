"""
Response curve fitting for Marketing Mix Modeling using Hill equation.

This module provides the ResponseCurveFit class for fitting saturation curves
to the relationship between media spend/impressions and predicted outcomes.

The implementation follows modern Python standards:
- Type hints for all parameters and return values
- Private methods prefixed with underscore (_)
- Keyword-only arguments for better API clarity
- Comprehensive docstrings in NumPy style
- Backward compatibility maintained for legacy code
"""

from typing import Optional, Literal, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from tqdm import tqdm

import logging
logger = logging.getLogger('deepcausalmmm')


class ResponseCurveFit:
    """
    Fit Hill equation response curves to marketing mix model predictions.
    
    The Hill equation models saturation effects:
    y = bottom + (top - bottom) * x^slope / (saturation^slope + x^slope)
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns: 'week_monday', 'spend', 'impressions', 'predicted'
        For DMA-level: also needs 'dmacode' column
    bottom_param : bool, default=False
        Whether to fit a non-zero intercept (bottom parameter)
        For MMM, typically False (response at zero spend = 0)
    Modellevel : str, default='Overall'
        'Overall': Single aggregated curve across all regions
        'DMA': Separate curves for each DMA
    Datecol : str, default='week_monday'
        Name of the date column
    
    Attributes
    ----------
    top : float
        Maximum response (saturation level)
    bottom : float
        Minimum response (typically 0)
    saturation : float
        Spend level at half-maximum response
    slope : float
        Steepness of the curve
    r_2 : float
        R-squared score of the fitted curve
    equation : str
        String representation of the fitted equation
    figure : go.Figure
        Plotly figure object (if generate_figure=True)
    
    Examples
    --------
    >>> # Prepare data
    >>> data = pd.DataFrame({
    ...     'week_monday': dates,
    ...     'spend': spend_values,
    ...     'impressions': impression_values,
    ...     'predicted': model_predictions
    ... })
    >>> 
    >>> # Fit overall response curve
    >>> fitter = ResponseCurveFit(data, Modellevel='Overall')
    >>> fitter.fit_model(
    ...     title="Response Curve",
    ...     x_label="Impressions",
    ...     y_label="Predicted Visits",
    ...     generate_figure=True,
    ...     save_figure=True,
    ...     output_path='response_curve.html'
    ... )
    >>> print(f"R²: {fitter.r_2:.3f}")
    >>> print(f"Slope: {fitter.slope:.3f}")
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        *,
        bottom_param: bool = False,
        model_level: Literal['Overall', 'DMA'] = 'Overall',
        date_col: str = 'week_monday'
    ) -> None:
        """
        Initialize ResponseCurveFit.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with required columns
        bottom_param : bool, default=False
            Whether to fit non-zero intercept
        model_level : {'Overall', 'DMA'}, default='Overall'
            Aggregation level for fitting
        date_col : str, default='week_monday'
            Name of date column
        """
        self.data = data
        self.bottom_param = bottom_param
        self.model_level = model_level
        self.date_col = date_col
        
        # Backward compatibility
        self.Modellevel = model_level
        self.Datecol = date_col

    def _hill_equation(self, X: np.ndarray, *params) -> np.ndarray:
        """
        Hill equation for saturation modeling.
        
        Parameters
        ----------
        X : np.ndarray
            Input values (spend/impressions)
        params : tuple
            (top, bottom, saturation, slope)
        
        Returns
        -------
        np.ndarray
            Predicted response values
        """
        self.top = params[0]
        self.bottom = params[1] if self.bottom_param else 0
        self.saturation = params[2]
        self.slope = params[3]

        return self.bottom + (self.top - self.bottom) * X**self.slope / (
            self.saturation**self.slope + X**self.slope
        )
    
    # Backward compatibility
    def Hill(self, X: np.ndarray, *params) -> np.ndarray:
        """Backward compatibility wrapper for _hill_equation."""
        return self._hill_equation(X, *params)
    
    def _get_initial_params(self, curve_fit_kws: dict) -> dict:
        """
        Get initial parameter guesses and bounds.
        
        Parameters
        ----------
        curve_fit_kws : dict
            Additional keyword arguments for scipy.optimize.curve_fit
        
        Returns
        -------
        dict
            Updated curve_fit_kws with p0 and bounds
        """
        min_data = np.amin(self._y_data)
        max_data = np.amax(self._y_data)

        h = abs(max_data - min_data)
        param_initial = [max_data, min_data, 0.5 * (self._X_data[-1] - self._X_data[0]), 1]
        param_bounds = (
            [max_data - 0.5 * h, min_data - 0.5 * h, self._X_data[0] * 0.1, 0.01],
            [max_data + 0.5 * h, min_data + 0.5 * h, self._X_data[-1] * 10, 100],
        )
        curve_fit_kws.setdefault("p0", param_initial)
        curve_fit_kws.setdefault("bounds", param_bounds)
        return curve_fit_kws
    
    def _fit_curve(self, curve_fit_kws: dict) -> List[float]:
        """
        Fit the Hill curve to data.
        
        Parameters
        ----------
        curve_fit_kws : dict
            Keyword arguments for scipy.optimize.curve_fit
        
        Returns
        -------
        list
            Fitted parameters [top, bottom, saturation, slope]
        """
        curve_fit_kws = self._get_initial_params(curve_fit_kws)
        popt, _ = curve_fit(self._hill_equation, self._X_data, self._y_data, **curve_fit_kws)
        if not self.bottom_param:
            popt[1] = 0
        return [float(param) for param in popt]
    
    # Backward compatibility
    def get_param(self, curve_fit_kws: dict) -> List[float]:
        """Backward compatibility wrapper for _fit_curve."""
        return self._fit_curve(curve_fit_kws)

    def _calculate_r2_and_plot(
        self,
        x_fit: np.ndarray,
        y_fit: np.ndarray,
        x_label: str,
        y_label: str,
        title: str,
        sigfigs: int,
        log_x: bool,
        print_r_sqr: bool,
        generate_figure: bool,
        view_figure: bool,
        *params,
    ) -> None:
        """
        Calculate R² and generate visualization.
        
        Parameters
        ----------
        x_fit : np.ndarray
            X values for fitted curve
        y_fit : np.ndarray
            Y values for fitted curve
        x_label : str
            X-axis label
        y_label : str
            Y-axis label
        title : str
            Plot title
        sigfigs : int
            Significant figures for equation display
        log_x : bool
            Whether to use log scale for x-axis
        print_r_sqr : bool
            Whether to print R² score
        generate_figure : bool
            Whether to generate visualization
        view_figure : bool
            Whether to display the figure
        params : tuple
            Fitted parameters
        """
        corrected_y_data = self._hill_equation(self._X_data, *params)
        self.r_2 = r2_score(self._y_data, corrected_y_data)

        if generate_figure:
            self.figure = go.Figure()
            
            self.figure.add_trace(go.Scatter(
                x=self._X_data, y=self._y_data,
                name='Observed Data',
                mode='markers'
            ))
            
            self.figure.add_trace(go.Scatter(
                x=x_fit, y=y_fit,
                name='Fitted Model',
                mode='lines'
            ))
            
            self.figure.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                legend_title="Legend",
                width=1500,
                height=900,
                yaxis_zeroline=False,
                xaxis_zeroline=False
            )
        
            if print_r_sqr:
                logger.info(f"   R² Score: {self.r_2:.4f}")

            if view_figure:
                self.figure.show()
    
    # Backward compatibility
    def regression(self, x_fit, y_fit, x_label, y_label, title, sigfigs, log_x, print_r_sqr, generate_figure, view_figure, *params) -> None:
        """Backward compatibility wrapper for _calculate_r2_and_plot."""
        return self._calculate_r2_and_plot(x_fit, y_fit, x_label, y_label, title, sigfigs, log_x, print_r_sqr, generate_figure, view_figure, *params)

    def fit(
        self,
        *,
        x_label: str = "x",
        y_label: str = "y",
        title: str = "Fitted Hill equation",
        sigfigs: int = 6,
        log_x: bool = False,
        print_r_sqr: bool = True,
        generate_figure: bool = True,
        view_figure: bool = False,
        save_figure: bool = False,
        output_path: Optional[str] = None,
        curve_fit_kws: Optional[dict] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fit Hill equation to the data.
        
        Parameters
        ----------
        x_label : str, default='x'
            X-axis label
        y_label : str, default='y'
            Y-axis label
        title : str, default='Fitted Hill equation'
            Plot title
        sigfigs : int, default=6
            Significant figures for equation display
        log_x : bool, default=False
            Whether to use log scale for x-axis
        print_r_sqr : bool, default=True
            Whether to print R² score
        generate_figure : bool, default=True
            Whether to generate visualization
        view_figure : bool, default=False
            Whether to display the figure
        save_figure : bool, default=False
            Whether to save the figure
        output_path : str, optional
            Path to save the figure (if save_figure=True)
        curve_fit_kws : dict, optional
            Additional keyword arguments for scipy.optimize.curve_fit
        
        Returns
        -------
        pd.DataFrame or None
            For DMA-level: DataFrame with parameters for each DMA
            For Overall: None (parameters stored as attributes)
        """
        if self.model_level == 'Overall':
            cpi = self.data['spend'].sum() / self.data['impressions'].sum()
            
            self.data_agg = self.data[[self.date_col, 'impressions', 'predicted']].groupby(self.date_col).sum()
            self.data_agg['spend'] = self.data_agg['impressions'] * cpi
            self.data_agg.sort_values(by='spend', inplace=True)
            
            self._X_data = np.array(self.data_agg['spend'])
            self._y_data = np.array(self.data_agg['predicted'])
        
            if self._X_data[0] > self._X_data[-1]:
                raise ValueError(
                    f"The first point {self._X_data[0]} and the last point {self._X_data[-1]} are not amenable with the scipy.curvefit function."
                )

            if curve_fit_kws is None:
                curve_fit_kws = {}
            
            self.generate_figure = generate_figure
            self.x_fit = np.logspace(
                np.log10(self._X_data[0]), np.log10(self._X_data[-1]), len(self._y_data)
            )
            
            self.fit_flag = False
            
            try:
                params = self._fit_curve(curve_fit_kws)
                
                self.y_fit = self._hill_equation(self.x_fit, *params)
                self.equation = f"{np.round(self.bottom, sigfigs)} + ({np.round(self.top, sigfigs)}-{np.round(self.bottom, sigfigs)})*x**{(np.round(self.slope, sigfigs))} / ({np.round(self.saturation, sigfigs)}**{(np.round(self.slope, sigfigs))} + x**{(np.round(self.slope, sigfigs))})"

                self._calculate_r2_and_plot(
                    self.x_fit,
                    self.y_fit,
                    x_label,
                    y_label,
                    title,
                    sigfigs,
                    log_x,
                    print_r_sqr,
                    generate_figure,
                    view_figure,
                    *params,
                )
                
                if save_figure and output_path and generate_figure:
                    self.figure.write_html(output_path)
                    logger.info(f"   Figure saved to: {output_path}")
                
                self.fit_flag = True
                
            except RuntimeError as re:
                self.r_2 = 0
                logger.warning(f"    Fitting failed: {re}")

        elif self.model_level == 'DMA':
            self.generate_figure = False
            self.view_figure = False
            self.print_r_sqr = False
            
            cpi = self.data[['dmacode', 'impressions', 'spend']].groupby('dmacode').sum().reset_index()
            
            cpi['cpi'] = cpi['spend'] / cpi['impressions']
            
            self.data_agg = self.data[['dmacode', self.date_col, 'impressions', 'predicted']].groupby(['dmacode', self.date_col]).sum()
            self.data_agg = self.data_agg.merge(cpi[['dmacode', 'cpi']], on='dmacode')
            
            self.data_agg['spend'] = self.data_agg['impressions'] * self.data_agg['cpi']
            self.data_agg.sort_values(by=['dmacode', 'spend'], inplace=True)
            
            dmas = self.data_agg['dmacode'].unique()
            
            bottom = []
            top = []
            slope = []
            saturation = []
            r_2 = []
            
            for i in tqdm(dmas, desc='Running...'):
                self._X_data = np.array(self.data_agg[self.data_agg['dmacode'] == i]['spend'])
                self._y_data = np.array(self.data_agg[self.data_agg['dmacode'] == i]['predicted'])
                
                if self._X_data[0] > self._X_data[-1]:
                    raise ValueError(
                        f"The first point {self._X_data[0]} and the last point {self._X_data[-1]} are not amenable with the scipy.curvefit function."
                    )
                
                curve_fit_kws = {}
                
                try:
                    params = self._fit_curve(curve_fit_kws)
                    
                    corrected_y_data = self._hill_equation(self._X_data, *params)
                    self.r_2 = r2_score(self._y_data, corrected_y_data)

                    bottom.append(self.bottom)
                    top.append(self.top)
                    slope.append(self.slope)
                    saturation.append(self.saturation)
                    r_2.append(self.r_2)
                
                except ValueError as e:
                    bottom.append(0)
                    top.append(0)
                    slope.append(0)
                    saturation.append(0)
                    r_2.append(0)
                    continue
                
                except RuntimeError as re:
                    bottom.append(0)
                    top.append(0)
                    slope.append(0)
                    saturation.append(0)
                    r_2.append(0)
                    
                    re = "for dmacode: " + str(i) + " " + str(re)
                    logger.warning(re)
                    continue
                
            return pd.DataFrame({
                'dmacode': list(dmas),
                'bottom': bottom,
                'top': top,
                'slope': slope,
                'saturation': saturation,
                'r_2': r_2
            })
    
    # Backward compatibility
    def fit_model(self, **kwargs) -> Optional[pd.DataFrame]:
        """Backward compatibility wrapper for fit()."""
        return self.fit(**kwargs)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict response for new spend levels (Overall level only).
        
        Parameters
        ----------
        X : np.ndarray
            New spend/impression values
        
        Returns
        -------
        np.ndarray
            Predicted response values
        """
        if self.fit_flag:
            return self.bottom + (self.top - self.bottom) * X**self.slope / (
                self.saturation**self.slope + X**self.slope
            )
    
    def get_summary(self):
        """
        Get summary of fitted parameters.
        
        Returns
        -------
        dict
            Dictionary with 'params', 'r2', and 'equation'
        """
        return {
            'params': {
                'top': self.top,
                'bottom': self.bottom,
                'saturation': self.saturation,
                'slope': self.slope
            },
            'r2': self.r_2,
            'equation': self.equation if hasattr(self, 'equation') else None
        }


# Backward compatibility alias
ResponseCurveFitter = ResponseCurveFit
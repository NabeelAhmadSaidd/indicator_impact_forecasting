$categories = @('commodity', 'crypto', 'equity', 'forex', 'index')
$assets = @{
'commodity' = @('crudeoil', 'gold')
'crypto' = @('bitcoin', 'ethereum')
'equity' = @('apple', 'microsoft')
'forex' = @('eurusd', 'usdjpy')
'index' = @('nasdaq', 'sp500')
}
$groups = @('group0_ohlcv', 'group1_momentum', 'group2_trend', 'group3_volatility', 'group4_all')
$models = @('lstm', 'gru')
$seeds = @(42, 123, 456, 739, 1126)
foreach ($category in $categories) {
  foreach ($asset in $assets[$category]) {
    foreach ($group in $groups) {
      foreach ($model in $models) {
        foreach ($seed in $seeds) {
          $path = "models/$category/$asset/$group/$model/$seed"
          New-Item -ItemType Directory -Path $path -Force
        }
      }
    }
  }
}
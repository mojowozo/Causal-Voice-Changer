using Microsoft.UI.Xaml;

namespace CausalVC.Realtime;

public partial class App : Application
{
    MainWindow? _window;

    public App() => InitializeComponent();

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        _window = new MainWindow();
        _window.Activate();
    }
}

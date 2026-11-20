namespace CausalVC.Training.Ui.Pages;

/// This version and updates (pages.rs: the About page).
public sealed class AboutPage : PageBase
{
    public AboutPage(IHost host, SettingsSection updates) : base(host)
    {
        Column.Children.Add(updates);
    }
}

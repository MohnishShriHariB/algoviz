# Generated by Django 4.2.5 on 2023-09-27 12:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('algoviz', '0004_delete_algo'),
    ]

    operations = [
        migrations.AddField(
            model_name='pics',
            name='gif',
            field=models.ImageField(default='C:\\Users\\Mohnish\\Desktop\\algo_viz-project\\midea\\algoviz\\images\\frame0_4nL03ZR', upload_to='images/'),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='pics',
            name='frame',
            field=models.ImageField(upload_to='images/'),
        ),
    ]

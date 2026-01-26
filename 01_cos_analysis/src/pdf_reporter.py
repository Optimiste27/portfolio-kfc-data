"""
Module de génération de rapports PDF automatisés pour l'analyse COS
VERSION CORRIGÉE - Bug mensuel fixé + warnings pandas résolus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import matplotlib
matplotlib.use('Agg')  # Pour éviter les problèmes d'affichage

class COSPDFReporter:
    """Générateur de rapports PDF pour l'analyse COS - VERSION CORRIGÉE"""
    
    def __init__(self, output_dir="reports"):
        """Initialise le générateur de rapports"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Styles pour le PDF
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Crée des styles personnalisés pour le PDF"""
        # Style pour le titre principal
        self.styles.add(ParagraphStyle(
            name='MainTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2E86AB'),
            alignment=TA_CENTER,
            spaceAfter=30
        ))
        
        # Style pour les sous-titres
        self.styles.add(ParagraphStyle(
            name='SubTitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#A23B72'),
            spaceAfter=20
        ))
        
        # Style pour le texte normal
        self.styles.add(ParagraphStyle(
            name='NormalText',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14
        ))
        
        # Style pour les métriques importantes
        self.styles.add(ParagraphStyle(
            name='Metric',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#2E86AB'),
            alignment=TA_CENTER,
            spaceAfter=10
        ))
    
    def _safe_date_conversion(self, df):
        """Convertit la colonne date en sécurité (évite SettingWithCopyWarning)"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def generate_weekly_report(self, transactions_df, restaurant_info_df, week_number=None):
        """
        Génère un rapport hebdomadaire PDF
        
        Args:
            transactions_df: DataFrame des transactions
            restaurant_info_df: DataFrame des informations restaurants
            week_number: Numéro de semaine (par défaut: semaine courante)
        
        Returns:
            str: Chemin du fichier PDF généré
        """
        print(f"📊 Génération du rapport hebdomadaire...")
        
        # Détermine la semaine
        if week_number is None:
            today = datetime.now()
            week_number = today.isocalendar()[1]
        
        # Convertit les dates en sécurité
        week_data = self._safe_date_conversion(transactions_df)
        
        # Filtre pour la semaine demandée
        week_data = week_data[
            week_data['date'].dt.isocalendar().week == week_number
        ]
        
        if week_data.empty:
            print(f"⚠️  Aucune donnée pour la semaine {week_number}")
            return None
        
        # Crée le nom du fichier
        filename = f"cos_weekly_report_w{week_number:02d}_{datetime.now().strftime('%Y%m%d')}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        # Crée le document PDF
        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Éléments du document
        story = []
        
        # 1. EN-TÊTE
        story.append(Paragraph(f"Rapport Hebdomadaire COS - Semaine {week_number}", self.styles['MainTitle']))
        story.append(Paragraph(f"Période: {week_data['date'].min().strftime('%d/%m/%Y')} - {week_data['date'].max().strftime('%d/%m/%Y')}", self.styles['SubTitle']))
        story.append(Paragraph(f"Généré le: {datetime.now().strftime('%d/%m/%Y %H:%M')}", self.styles['NormalText']))
        story.append(Spacer(1, 20))
        
        # 2. MÉTRIQUES GLOBALES
        story.append(Paragraph("📈 MÉTRIQUES GLOBALES", self.styles['SubTitle']))
        
        # Calcul des métriques
        total_revenue = week_data['revenue'].sum()
        avg_gap = week_data['gap_percentage'].mean()
        total_waste_cost = (week_data['waste_kg'] * week_data['theoretical_unit_cost']).sum()
        anomaly_rate = week_data['operational_anomaly'].mean() * 100
        
        # Tableau des métriques
        metrics_data = [
            ["Métrique", "Valeur", "Cible", "Statut"],
            ["Chiffre d'affaires", f"{total_revenue:,.0f} €", "N/A", "✅"],
            ["Écart COS moyen", f"{avg_gap:.2f}%", "< 4.0%", "✅" if avg_gap < 4.0 else "⚠️" if avg_gap < 6.0 else "❌"],
            ["Coût du gaspillage", f"{total_waste_cost:,.0f} €", "Minimiser", "✅" if total_waste_cost < 5000 else "⚠️"],
            ["Taux d'anomalies", f"{anomaly_rate:.1f}%", "< 15%", "✅" if anomaly_rate < 15 else "⚠️"]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[3*cm, 3*cm, 3*cm, 2*cm])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # 3. PERFORMANCE PAR RESTAURANT
        story.append(Paragraph("🏪 PERFORMANCE PAR RESTAURANT", self.styles['SubTitle']))
        
        # Top 5 restaurants avec plus d'écarts
        restaurant_perf = week_data.groupby('restaurant_name').agg({
            'gap_percentage': ['mean', 'count'],
            'revenue': 'sum',
            'operational_anomaly': 'mean'
        }).round(2)
        
        restaurant_perf.columns = ['Écart moyen %', 'Nb transactions', 'CA total', 'Taux anomalies']
        restaurant_perf_sorted = restaurant_perf.sort_values('Écart moyen %', ascending=False)
        
        # Tableau restaurants
        rest_data = [["Restaurant", "Écart moyen", "CA", "Anomalies", "Statut"]]
        for idx, row in restaurant_perf_sorted.head(10).iterrows():
            status = "✅" if row['Écart moyen %'] < 4.0 else "⚠️" if row['Écart moyen %'] < 6.0 else "❌"
            rest_data.append([
                idx[:20] + "..." if len(idx) > 20 else idx,
                f"{row['Écart moyen %']:.1f}%",
                f"{row['CA total']:,.0f} €",
                f"{row['Taux anomalies']*100:.1f}%",
                status
            ])
        
        rest_table = Table(rest_data, colWidths=[4*cm, 2.5*cm, 3*cm, 2.5*cm, 1.5*cm])
        rest_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#A23B72')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        
        story.append(rest_table)
        story.append(Spacer(1, 20))
        
        # 4. PRODUITS À RISQUE
        story.append(Paragraph("🍗 PRODUITS À RISQUE", self.styles['SubTitle']))
        
        product_perf = week_data.groupby('product_family').agg({
            'gap_percentage': 'mean',
            'waste_kg': 'sum',
            'revenue': 'sum'
        }).round(2)
        
        product_perf_sorted = product_perf.sort_values('gap_percentage', ascending=False)
        
        prod_data = [["Produit", "Écart moyen", "Gaspillage", "CA"]]
        for idx, row in product_perf_sorted.head(8).iterrows():
            prod_data.append([
                idx,
                f"{row['gap_percentage']:.1f}%",
                f"{row['waste_kg']:.1f} kg",
                f"{row['revenue']:,.0f} €"
            ])
        
        prod_table = Table(prod_data, colWidths=[4*cm, 3*cm, 3*cm, 3*cm])
        prod_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F18F01')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFF8E7')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        
        story.append(prod_table)
        story.append(Spacer(1, 20))
        
        # 5. RECOMMANDATIONS
        story.append(Paragraph("🎯 RECOMMANDATIONS PRIORITAIRES", self.styles['SubTitle']))
        
        # Analyse pour recommandations
        critical_restaurants = restaurant_perf_sorted[restaurant_perf_sorted['Écart moyen %'] > 6.0]
        critical_products = product_perf_sorted[product_perf_sorted['gap_percentage'] > 5.0]
        
        recommendations = []
        
        if not critical_restaurants.empty:
            worst_rest = critical_restaurants.index[0]
            worst_gap = critical_restaurants.iloc[0]['Écart moyen %']
            recommendations.append(f"🔴 <b>{worst_rest}</b>: Écart de {worst_gap:.1f}% - Audit recommandé")
        
        if not critical_products.empty:
            worst_prod = critical_products.index[0]
            worst_prod_gap = critical_products.iloc[0]['gap_percentage']
            recommendations.append(f"🍗 <b>{worst_prod}</b>: Écart de {worst_prod_gap:.1f}% - Formation équipe")
        
        if avg_gap > 4.0:
            recommendations.append(f"📈 <b>Écart global élevé</b>: {avg_gap:.1f}% - Renforcer contrôles")
        
        if len(recommendations) == 0:
            recommendations.append("✅ <b>Tous les indicateurs sont dans les cibles</b> - Maintenir les efforts")
        
        for rec in recommendations:
            story.append(Paragraph(rec, self.styles['NormalText']))
            story.append(Spacer(1, 8))
        
        story.append(Spacer(1, 20))
        
        # 6. PIED DE PAGE
        story.append(Paragraph("_" * 80, self.styles['NormalText']))
        footer_text = f"Rapport généré automatiquement - Système d'analyse COS KFC - Page 1"
        story.append(Paragraph(footer_text, ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )))
        
        # Génère le PDF
        doc.build(story)
        
        print(f"✅ Rapport généré: {filepath}")
        return filepath
    
    def generate_monthly_report(self, transactions_df, restaurant_info_df, month=None, year=None):
        """
        Génère un rapport mensuel PDF complet - VERSION CORRIGÉE
        """
        print(f"📅 Génération du rapport mensuel...")
        
        # Détermine le mois
        if month is None:
            month = datetime.now().month
        if year is None:
            year = datetime.now().year
        
        # Filtre les données du mois (copie pour éviter le warning)
        month_data = self._safe_date_conversion(transactions_df)
        month_data = month_data[
            (month_data['date'].dt.month == month) & 
            (month_data['date'].dt.year == year)
        ]
        
        if month_data.empty:
            print(f"⚠️  Aucune donnée pour {month}/{year}")
            return None
        
        # Nom du fichier
        month_names = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
                       "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
        month_name = month_names[month-1] if 1 <= month <= 12 else f"Mois {month}"
        
        filename = f"cos_monthly_report_{year}_{month:02d}_{month_name}_{datetime.now().strftime('%Y%m%d')}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        # Crée le document PDF spécifique pour le mois
        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        story = []
        
        # EN-TÊTE SPÉCIFIQUE MOIS
        story.append(Paragraph(f"Rapport Mensuel COS - {month_name} {year}", self.styles['MainTitle']))
        story.append(Paragraph(f"Période: {month_data['date'].min().strftime('%d/%m/%Y')} - {month_data['date'].max().strftime('%d/%m/%Y')}", self.styles['SubTitle']))
        story.append(Paragraph(f"Généré le: {datetime.now().strftime('%d/%m/%Y %H:%M')}", self.styles['NormalText']))
        story.append(Spacer(1, 20))
        
        # MÉTRIQUES MENSUELLES
        story.append(Paragraph("📈 MÉTRIQUES MENSUELLES", self.styles['SubTitle']))
        
        # Calcul des métriques mensuelles
        total_revenue = month_data['revenue'].sum()
        avg_gap = month_data['gap_percentage'].mean()
        total_waste_cost = (month_data['waste_kg'] * month_data['theoretical_unit_cost']).sum()
        anomaly_rate = month_data['operational_anomaly'].mean() * 100
        nb_transactions = len(month_data)
        
        # Tableau des métriques mensuelles
        metrics_data = [
            ["Métrique", "Valeur", "Cible", "Statut"],
            ["Transactions", f"{nb_transactions:,}", "N/A", "✅"],
            ["Chiffre d'affaires", f"{total_revenue:,.0f} €", "N/A", "✅"],
            ["Écart COS moyen", f"{avg_gap:.2f}%", "< 4.0%", "✅" if avg_gap < 4.0 else "⚠️" if avg_gap < 6.0 else "❌"],
            ["Coût du gaspillage", f"{total_waste_cost:,.0f} €", "Minimiser", "✅" if total_waste_cost/total_revenue*100 < 2 else "⚠️"],
            ["Taux d'anomalies", f"{anomaly_rate:.1f}%", "< 15%", "✅" if anomaly_rate < 15 else "⚠️"]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[3*cm, 3*cm, 3*cm, 2*cm])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # ÉVOLUTION HEBDOMADAIRE DANS LE MOIS
        story.append(Paragraph("📅 ÉVOLUTION HEBDOMADAIRE", self.styles['SubTitle']))
        
        # Calcule les métriques par semaine dans le mois
        month_data['week'] = month_data['date'].dt.isocalendar().week
        weekly_stats = month_data.groupby('week').agg({
            'gap_percentage': 'mean',
            'revenue': 'sum',
            'operational_anomaly': 'mean'
        }).round(2).reset_index()
        
        if not weekly_stats.empty:
            weekly_data = [["Semaine", "Écart moyen %", "CA €", "Anomalies %"]]
            for _, row in weekly_stats.iterrows():
                weekly_data.append([
                    f"Semaine {int(row['week'])}",
                    f"{row['gap_percentage']:.1f}%",
                    f"{row['revenue']:,.0f}",
                    f"{row['operational_anomaly']*100:.1f}%"
                ])
            
            weekly_table = Table(weekly_data, colWidths=[3*cm, 3*cm, 4*cm, 3*cm])
            weekly_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#A23B72')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ]))
            
            story.append(weekly_table)
        else:
            story.append(Paragraph("Aucune donnée hebdomadaire disponible", self.styles['NormalText']))
        
        story.append(Spacer(1, 20))
        
        # PERFORMANCE PAR RESTAURANT (Mensuelle)
        story.append(Paragraph("🏪 PERFORMANCE MENSUELLE PAR RESTAURANT", self.styles['SubTitle']))
        
        restaurant_perf = month_data.groupby('restaurant_name').agg({
            'gap_percentage': 'mean',
            'revenue': 'sum',
            'operational_anomaly': 'mean'
        }).round(2)
        
        restaurant_perf_sorted = restaurant_perf.sort_values('gap_percentage', ascending=False)
        
        rest_data = [["Restaurant", "Écart moyen", "CA", "Anomalies", "Statut"]]
        for idx, row in restaurant_perf_sorted.head(8).iterrows():
            status = "✅" if row['gap_percentage'] < 4.0 else "⚠️" if row['gap_percentage'] < 6.0 else "❌"
            rest_data.append([
                idx[:18] + "..." if len(idx) > 18 else idx,
                f"{row['gap_percentage']:.1f}%",
                f"{row['revenue']:,.0f} €",
                f"{row['operational_anomaly']*100:.1f}%",
                status
            ])
        
        rest_table = Table(rest_data, colWidths=[4*cm, 2.5*cm, 3*cm, 2.5*cm, 1.5*cm])
        rest_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3D5A80')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        
        story.append(rest_table)
        story.append(Spacer(1, 20))
        
        # RECOMMANDATIONS MENSUELLES
        story.append(Paragraph("🎯 RECOMMANDATIONS POUR LE MOIS", self.styles['SubTitle']))
        
        recommendations = []
        
        if avg_gap > 4.0:
            recommendations.append(f"📈 <b>Écart mensuel élevé</b>: {avg_gap:.1f}% - Objectif non atteint")
            recommendations.append("• Renforcer les contrôles qualité")
            recommendations.append("• Former les équipes sur les points critiques")
        
        if anomaly_rate > 15:
            recommendations.append(f"⚠️  <b>Taux d'anomalies élevé</b>: {anomaly_rate:.1f}%")
            recommendations.append("• Analyser les causes racines des anomalies")
            recommendations.append("• Mettre en place des actions correctives")
        
        waste_percentage = (total_waste_cost / total_revenue) * 100 if total_revenue > 0 else 0
        if waste_percentage > 2:
            recommendations.append(f"🗑️  <b>Gaspillage élevé</b>: {waste_percentage:.1f}% du CA")
            recommendations.append("• Optimiser les quantités préparées")
            recommendations.append("• Améliorer la rotation des stocks")
        
        if not recommendations:
            recommendations.append("✅ <b>Mois satisfaisant</b> - Tous les indicateurs dans les cibles")
            recommendations.append("• Maintenir les bonnes pratiques")
            recommendations.append("• Partager les succès avec les équipes")
        
        for rec in recommendations:
            story.append(Paragraph(rec, self.styles['NormalText']))
            story.append(Spacer(1, 5))
        
        # PIED DE PAGE
        story.append(Spacer(1, 20))
        story.append(Paragraph("_" * 80, self.styles['NormalText']))
        footer_text = f"Rapport Mensuel COS - {month_name} {year} - Page 1"
        story.append(Paragraph(footer_text, ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )))
        
        # Génère le PDF
        doc.build(story)
        
        print(f"✅ Rapport mensuel généré: {filepath}")
        return filepath
    
    def generate_restaurant_specific_report(self, transactions_df, restaurant_name):
        """
        Génère un rapport spécifique pour un restaurant
        """
        print(f"🏪 Génération du rapport pour {restaurant_name}...")
        
        # Filtre les données du restaurant (copie pour éviter le warning)
        rest_data = self._safe_date_conversion(transactions_df)
        rest_data = rest_data[rest_data['restaurant_name'] == restaurant_name]
        
        if rest_data.empty:
            print(f"⚠️  Aucune donnée pour {restaurant_name}")
            return None
        
        # Nom du fichier
        safe_name = restaurant_name.replace(" ", "_").replace("/", "_")
        filename = f"cos_report_{safe_name}_{datetime.now().strftime('%Y%m%d')}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        # Crée le document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        story = []
        
        # En-tête
        story.append(Paragraph(f"Rapport COS - {restaurant_name}", self.styles['MainTitle']))
        story.append(Paragraph(f"Période: {rest_data['date'].min().strftime('%d/%m/%Y')} - {rest_data['date'].max().strftime('%d/%m/%Y')}", self.styles['SubTitle']))
        story.append(Spacer(1, 20))
        
        # Métriques spécifiques
        avg_gap = rest_data['gap_percentage'].mean()
        total_revenue = rest_data['revenue'].sum()
        total_waste = rest_data['waste_kg'].sum()
        anomaly_rate = rest_data['operational_anomaly'].mean() * 100
        
        story.append(Paragraph("📊 PERFORMANCE DU RESTAURANT", self.styles['SubTitle']))
        
        metrics_data = [
            ["Métrique", "Valeur", "Cible", "Statut"],
            ["Écart COS moyen", f"{avg_gap:.2f}%", "< 4.0%", "✅" if avg_gap < 4.0 else "⚠️" if avg_gap < 6.0 else "❌"],
            ["Chiffre d'affaires", f"{total_revenue:,.0f} €", "N/A", "✅"],
            ["Gaspillage total", f"{total_waste:.1f} kg", "Minimiser", "✅" if total_waste < 100 else "⚠️"],
            ["Taux d'anomalies", f"{anomaly_rate:.1f}%", "< 15%", "✅" if anomaly_rate < 15 else "⚠️"]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[3*cm, 3*cm, 3*cm, 2*cm])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 30))
        
        # PERFORMANCE PAR PRODUIT
        story.append(Paragraph("🍗 PERFORMANCE PAR PRODUIT", self.styles['SubTitle']))
        
        product_perf = rest_data.groupby('product_family').agg({
            'gap_percentage': 'mean',
            'waste_kg': 'sum',
            'revenue': 'sum'
        }).round(2)
        
        product_perf_sorted = product_perf.sort_values('gap_percentage', ascending=False)
        
        prod_data = [["Produit", "Écart moyen", "Gaspillage", "CA"]]
        for idx, row in product_perf_sorted.head(8).iterrows():
            prod_data.append([
                idx,
                f"{row['gap_percentage']:.1f}%",
                f"{row['waste_kg']:.1f} kg",
                f"{row['revenue']:,.0f} €"
            ])
        
        prod_table = Table(prod_data, colWidths=[4*cm, 3*cm, 3*cm, 3*cm])
        prod_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F18F01')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        
        story.append(prod_table)
        story.append(Spacer(1, 20))
        
        # RECOMMANDATIONS
        story.append(Paragraph("🎯 ACTIONS RECOMMANDÉES", self.styles['SubTitle']))
        
        if avg_gap > 6.0:
            story.append(Paragraph(f"🔴 <b>ACTION REQUISE</b>: Écart COS de {avg_gap:.1f}% dépasse le seuil critique de 6%", self.styles['NormalText']))
            story.append(Paragraph("• Audit complet des processus opérationnels", self.styles['NormalText']))
            story.append(Paragraph("• Formation de l'équipe sur les recettes standards", self.styles['NormalText']))
            story.append(Paragraph("• Vérification des équipements et pesées", self.styles['NormalText']))
        elif avg_gap > 4.0:
            story.append(Paragraph(f"⚠️  <b>AMÉLIORATION NÉCESSAIRE</b>: Écart COS de {avg_gap:.1f}% dépasse la cible de 4%", self.styles['NormalText']))
            story.append(Paragraph("• Analyse détaillée par famille de produits", self.styles['NormalText']))
            story.append(Paragraph("• Renforcement des contrôles qualité", self.styles['NormalText']))
            story.append(Paragraph("• Revue des procédures de gestion des stocks", self.styles['NormalText']))
        else:
            story.append(Paragraph(f"✅ <b>PERFORMANCE SATISFAISANTE</b>: Écart COS de {avg_gap:.1f}% dans les cibles", self.styles['NormalText']))
            story.append(Paragraph("• Maintenir les bonnes pratiques", self.styles['NormalText']))
            story.append(Paragraph("• Partager les réussites avec l'équipe", self.styles['NormalText']))
            story.append(Paragraph("• Continuer le monitoring régulier", self.styles['NormalText']))
        
        # PIED DE PAGE
        story.append(Spacer(1, 20))
        story.append(Paragraph("_" * 80, self.styles['NormalText']))
        footer_text = f"Rapport Restaurant - {restaurant_name} - Page 1"
        story.append(Paragraph(footer_text, ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )))
        
        # Génère le PDF
        doc.build(story)
        
        print(f"✅ Rapport restaurant généré: {filepath}")
        return filepath

def test_pdf_generation():
    """Teste la génération de PDF"""
    print("🧪 Test de génération PDF...")
    
    # Crée des données de test
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    
    test_data = []
    for i in range(100):
        test_data.append({
            'date': np.random.choice(dates),
            'restaurant_name': np.random.choice(['KFC Paris Louvre', 'KFC Lyon Part-Dieu', 'KFC Marseille Vieux Port']),
            'product_family': np.random.choice(['Poulet Frit', 'Sandwichs', 'Accompagnements']),
            'gap_percentage': np.random.normal(4.0, 2.0),
            'revenue': np.random.uniform(100, 1000),
            'waste_kg': np.random.uniform(0, 5),
            'operational_anomaly': np.random.random() > 0.9,
            'theoretical_unit_cost': np.random.uniform(1.0, 5.0)
        })
    
    df = pd.DataFrame(test_data)
    restaurant_df = pd.DataFrame({
        'restaurant_name': ['KFC Paris Louvre', 'KFC Lyon Part-Dieu', 'KFC Marseille Vieux Port'],
        'city': ['Paris', 'Lyon', 'Marseille']
    })
    
    # Test le générateur
    reporter = COSPDFReporter()
    
    # Génère différents rapports
    weekly = reporter.generate_weekly_report(df, restaurant_df, week_number=1)
    monthly = reporter.generate_monthly_report(df, restaurant_df, month=1, year=2024)
    specific = reporter.generate_restaurant_specific_report(df, 'KFC Paris Louvre')
    
    print(f"\n📁 Rapports générés:")
    print(f"   Hebdomadaire: {'✅' if weekly else '❌'}")
    print(f"   Mensuel: {'✅' if monthly else '❌'}")
    print(f"   Spécifique: {'✅' if specific else '❌'}")
    
    return weekly is not None

if __name__ == "__main__":
    # Teste la génération
    if test_pdf_generation():
        print("\n✅ Test PDF réussi !")
    else:
        print("\n❌ Test PDF échoué")